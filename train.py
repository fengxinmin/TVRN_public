import os
import math
import argparse
import random
import logging
import numpy as np
from einops import rearrange
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from data.util import rgb2ycbcr_tensor


def init_dist(backend='nccl', **kwargs):
    """Initialize distributed training environment."""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def setup_distributed(opt, args):
    """Configure distributed training settings."""
    if args.launcher == 'none':
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    return rank


def setup_logging_and_tb(opt, args, rank):
    """Setup loggers and tensorboard writers."""
    tb_logger = {}
    
    if rank <= 0:
        if not opt['path'].get('resume_state'):
            util.mkdir_and_rename(opt['path']['experiments_root'])
            util.mkdirs(opt['path']['experiments_root'])

        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], 
                          level=logging.INFO, screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], 
                          level=logging.INFO, screen=True, tofile=True)
        
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(f'Using tensorboardX for PyTorch {version}')
                from tensorboardX import SummaryWriter
            
            # Initialize empty dict for lazy loading
            pass
        else:
            logger = logging.getLogger('base')
    else:
        util.setup_logger('base', opt['path']['log'], 'train', 
                          level=logging.INFO, screen=True)
        logger = logging.getLogger('base')
        
    return logger, tb_logger


def create_datasets_and_loaders(opt, rank):
    """Create datasets and dataloaders based on configuration."""
    dataset_ratio = 200
    train_set = None
    train_loader = None
    val_loader = None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, torch.distributed.get_world_size(), 
                                                torch.distributed.get_rank(), dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
                
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler, shuffle=False)
            
            if rank <= 0:
                logger = logging.getLogger('base')
                logger.info(f'Number of train images: {len(train_set)}, iters: {train_size}')
                logger.info(f'Total epochs needed: {total_epochs} for iters {total_iters}')
                
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None, shuffle=False)
            if rank <= 0:
                logger = logging.getLogger('base')
                logger.info(f'Number of val images in [{dataset_opt["name"]}]: {len(val_set)}')
        else:
            raise NotImplementedError(f'Phase [{phase}] is not recognized.')
            
    assert train_loader is not None
    return train_set, train_loader, val_loader, total_epochs


def run_validation(model, val_loader, opt, current_step, tb_logger, rank, args):
    """Execute validation loop and compute metrics."""
    avg_psnr, avg_ssim = 0.0, 0.0
    avg_psnr_lr, avg_psnr_y, avg_ssim_y = 0.0, 0.0, 0.0
    avg_psnr_vfi, avg_psnr_lr_compress = 0.0, 0.0
    avg_bpp = 0.0
    idx = 0
    
    max_val_samples = 10
    
    with torch.no_grad():
        for val_id, val_data in enumerate(tqdm(val_loader)):
            if val_id >= max_val_samples:
                break
            
            model.feed_data(val_data)
            model.test()
            
            t_step = model.input.shape[0]
            idx += t_step
            
            gt_img = model.real_H
            vsr_img = model.fake_H
            lrgt_img = model.real_H[:, ::2]
            lr_img = model.vlr
            vfi_gt_img = model.real_H[:, 1::2]
            vfi_img = model.fake_H[:, 1::2]
            
            bpp = model.bpp.detach()
            if isinstance(bpp, list):
                avg_bpp += float(sum(bpp).item() / len(bpp))
            elif isinstance(bpp, torch.Tensor):
                avg_bpp += bpp.item()
            
            vsr_img = torch.clamp(vsr_img * 255, min=0, max=255)
            
            for i in range(t_step):
                # PSNR calculation
                mse = torch.mean((gt_img[i] - vsr_img[i]) ** 2)
                psnr = 20 * math.log10(255.0 / math.sqrt(mse))
                avg_psnr += psnr
                
                # SSIM calculation
                ssim_list = []
                for j in range(gt_img[i].shape[0]):
                    ssim_list.append(util.ms_ssim(
                        gt_img[i][j].unsqueeze(0).float(), 
                        vsr_img[i][j].unsqueeze(0), 
                        data_range=255.0
                    ).item())
                ssim = sum(ssim_list) / len(ssim_list)
                avg_ssim += ssim
                
                # Y-channel PSNR/SSIM
                sr_img_y = rgb2ycbcr_tensor(gt_img[i] * 255, only_y=True) / 255.
                gt_img_y = rgb2ycbcr_tensor(vsr_img[i] * 255, only_y=True) / 255.
                
                mse_y = torch.mean((gt_img_y - sr_img_y) ** 2)
                y_psnr = 20 * math.log10(255.0 / math.sqrt(mse_y))
                avg_psnr_y += y_psnr
                
                ssim_y_list = []
                for j in range(gt_img[i].shape[0]):
                    ssim_y_list.append(util.ms_ssim(
                        gt_img_y[j].unsqueeze(0), 
                        sr_img_y[j].unsqueeze(0), 
                        data_range=255.0
                    ).item())
                y_ssim = sum(ssim_y_list) / len(ssim_y_list)
                avg_ssim_y += y_ssim
                
                # Placeholder metrics (initialized to 0)
                psnr_lr, psnr_vfi, psnr_lr_compress = 0, 0, 0
                avg_psnr_lr += psnr_lr
                avg_psnr_vfi += psnr_vfi
                
                if opt.get('codec', False):
                    avg_psnr_lr_compress += psnr_lr_compress

    # Normalize averages
    avg_psnr /= idx
    avg_psnr_lr /= idx
    avg_psnr_y /= idx
    avg_ssim_y /= idx
    avg_psnr_vfi /= idx
    if opt.get('codec', False):
        avg_psnr_lr_compress /= idx
        avg_bpp /= idx
    
    # Logging
    logger = logging.getLogger('base')
    logger.info(f'# Validation # PSNR: {avg_psnr:.4f}, LR PSNR: {avg_psnr_lr:.4f}, '
                f'LR_COMP PSNR: {avg_psnr_lr_compress:.4f}, Y-PSNR: {avg_psnr_y:.4f}, '
                f'Y-SSIM: {avg_ssim_y:.4f}, PSNR-VFI: {avg_psnr_vfi:.4f}, bpp: {avg_bpp:.4f}')
    
    logger_val = logging.getLogger('val')
    logger_val.info(f'<epoch:{epoch}, iter:{current_step}> psnr: {avg_psnr:.4f}, '
                    f'LR psnr: {avg_psnr_lr:.4f}, LR_compress psnr: {avg_psnr_lr_compress:.4f}, '
                    f'bpp: {avg_bpp:.4f}')
    
    # TensorBoard logging
    if opt['use_tb_logger'] and 'debug' not in opt['name'] and rank <= 0:
        metric_mappings = [
            ('val_psnr/avg-psnr', avg_psnr),
            ('val_psnr/avg-ssim', avg_ssim),
            ('val_psnrpsnr/vfi-psnr', avg_psnr_vfi),
            ('val_psnrpsnr/y-psnr', avg_psnr_y),
            ('val_psnrpsnr/vlr-psnr', avg_psnr_lr),
            ('val_psnrpsnr/vlr_compress-psnr', avg_psnr_lr_compress),
            ('val_bpp', avg_bpp)
        ]
        
        for key, value in metric_mappings:
            if key not in tb_logger:
                path = args.tensorboard_dir + key.replace('/', '_')
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tb_logger[key] = SummaryWriter(log_dir=path)
                except ImportError:
                    from tensorboardX import SummaryWriter
                    tb_logger[key] = SummaryWriter(log_dir=path)
            
            tb_logger[key].add_scalar('validation', value, current_step)

    model.empty_cache()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tensorboard_dir', type=str, default='/output/events/')
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=True)
    
    # Distributed Setup
    rank = setup_distributed(opt, args)
    
    # Resume State
    resume_state = None
    start_epoch = 0
    current_step = 0
    
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])
    else:
        resume_state = None
        
    # Logging Setup
    logger, tb_logger = setup_logging_and_tb(opt, args, rank)
    opt = option.dict_to_nonedict(opt)
    
    # Random Seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info(f'Random seed: {seed}')
    util.set_random_seed(seed)
    
    torch.backends.cudnn.benchmark = True
    
    # Data Loading
    train_set, train_loader, val_loader, total_epochs = create_datasets_and_loaders(opt, rank)
    
    # Model Creation
    model = create_model(opt)
    
    # Resume Training
    if resume_state:
        logger.info(f'Resuming training from epoch: {resume_state["epoch"]}, iter: {resume_state["iter"]}.')
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0
        
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_step}')
    
    torch.autograd.set_detect_anomaly(False)
    
    # Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
                
            inplace_flag = True
            model.feed_data(train_data)
            
            if opt['train'].get('finetune_restoration', False):
                model.optimize_parameters_restoration(inplace_flag=inplace_flag, current_step=current_step)
            else:
                model.optimize_parameters(inplace_flag=inplace_flag, current_step=current_step)
                
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            
            # Logging
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = f'<iter:{current_step:8,d}, lr:{model.get_current_learning_rate():.2e}> '
                
                for k, v in logs['train_loss'].items():
                    mean_val = sum(v)/len(v)
                    message += f'{k}: {mean_val:.2e} '
                    
                    if opt['use_tb_logger'] and 'debug' not in opt['name'] and rank <= 0:
                        if k not in tb_logger:
                            tb_logger[k] = SummaryWriter(log_dir=args.tensorboard_dir + k)
                        tb_logger[k].add_scalar('loss', mean_val, current_step)
                        
                if rank <= 0:
                    logger.info(message)
                model.log_dict['train_loss'] = None
            
            # Save Checkpoint
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
            
            # Validation
            if (current_step == 100 or current_step % opt['train']['val_freq'] == 0) and rank <= 0:
                run_validation(model, val_loader, opt, current_step, tb_logger, rank, args)
                
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()