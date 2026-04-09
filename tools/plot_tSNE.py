import os
import math
import argparse
import random
import logging
import numpy as np
from einops import rearrange
import matplotlib.font_manager as fm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from data.util import rgb2ycbcr_tensor
    
def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main(ranker_path=None):
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='/output/events' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    opt['datasets']['train']['out_triplet_qp'] = True
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, None, shuffle=False, pin_memory=opt['datasets']['train'])  # 指定sampler之后不能指定shuffle=True
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None, shuffle=False)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    opt['is_train'] = False
    opt['path']['pretrain_model_ranker'] = ranker_path
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
    
    
    features_list = []
    target_list = ['QP17', 'QP22', 'QP27', 'QP32', 'QP37']
    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > 5:
                break
            model.feed_data(train_data)
            features = model.plt_ranker(out_accu=False)
            if len(features_list) == 0:
                features_list = [a.detach().cpu().numpy() for a in features]
            else:
                for i in range(len(features_list)):
                    features_list[i] = np.concatenate([features_list[i], features[i].detach().cpu().numpy()], axis=0)
            model.empty_cache()
            torch.cuda.empty_cache()
    features = np.concatenate(features_list, axis=0)

    # 创建标签列表，对应每个样本的类别标签
    labels = []
    for i, features_tensor in enumerate(features_list):
        labels.extend([target_list[i]] * features_tensor.shape[0])


    return features, labels


if __name__ == '__main__':
    # 模型保存在/model/fengxm/VRN/Ranker/
    ranker_path_list = [
        "/model/fengxm/VRN/Ranker/Ranker_adj_double_l1.pth",
        "/model/fengxm/VRN/Ranker/Ranker_adj_double_rank.pth",
        "/model/fengxm/VRN/Ranker/Ranker_adj_double_rank_margin.pth",
        "/model/fengxm/VRN/Ranker/Ranker_adj_double_rank_margin_aux.path"
        # "/model/fengxm/VRN/MIMO_VRN/Ranker_v1_b400_adj_double_rank_margin_aux/30000_Ranker.pth"
    ]
    ranker_label_list = [
        r"(a) L1 loss", 
        r"(b) $\max\left(0, \left(s_i - s_j\right) \cdot \kappa + \xi\right)$", 
        r"(c) $\max\left(0, \left(s_i - s_j\right) \cdot \kappa + \xi(i,j)\right)$", 
        r"(d) $\max\left(0, \left(s_i - s_j\right) \cdot \kappa + \xi(i,j)\right) + \left\| s_i - s_k\right\|$"
    ]
    target_list = ['QP17', 'QP22', 'QP27', 'QP32', 'QP37']
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Adjusted figure size
    legend_loc = ['upper right', 'upper left', "lower right", "upper left"]
    for ranker_id, ranker_path in enumerate(ranker_path_list):
        features, labels = main(ranker_path=ranker_path)
        # 使用 t-SNE 降维
        if ranker_id == 3:
            n_iter = 10000
        else:
            n_iter = 5000
        tsne = TSNE(n_components=2, perplexity=20, learning_rate=200, n_iter=5000, random_state=42)
        features_tsne = tsne.fit_transform(features)

        # 绘制 t-SNE 结果
        row = ranker_id // 2
        col = ranker_id % 2

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        markers = ['o', 's', 'D', '^', 'P']  # 不同形状的marker

        for i, target in enumerate(np.unique(target_list)):
            # idx = np.where(labels == target)
            idx = [j for j, label in enumerate(labels) if label == target]
            # 为了让图（d）的QP顺序和其他几个一致，将y进行翻转
            if ranker_id == 3:
                axs[row, col].scatter(features_tsne[idx, 0], -features_tsne[idx, 1], c=colors[i], marker=markers[i], label=target, alpha=0.6)
            else:
                axs[row, col].scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=colors[i], marker=markers[i], label=target, alpha=0.6)

        # if ranker_id == 0:
            # axs[row, col].legend()
        
        axs[row, col].legend(loc=legend_loc[ranker_id],facecolor='white', framealpha=1)
        axs[row, col].set_title("")  # Clear default title

        # Set custom title below the subplot
        axs[row, col].text(0.5, -0.1, ranker_label_list[ranker_id], fontsize=14, ha='center', transform=axs[row, col].transAxes)

        axs[row, col].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                  labelbottom=False, labelleft=False)  # 去掉刻度线和刻度标签

    plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing between subplots
    plt.tight_layout()
    plt.savefig('/code/tsne_features_grid.png', dpi=300)
    plt.savefig('/code/tsne_features_grid.pdf', dpi=500)
    plt.show()