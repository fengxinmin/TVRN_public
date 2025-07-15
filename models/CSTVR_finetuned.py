import logging
from collections import OrderedDict
import cv2
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from einops import rearrange
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import torch.nn.functional as F
logger = logging.getLogger('base')
import random


class CSTVRModel(BaseModel):
    """Temporal video rescaling network

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self, opt):
        super(CSTVRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.color = opt['network_G']['color']  # 默认是RGB，其他选项YUV
        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.color =  opt['network_G']['color'] 

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)

        self.load()
        if self.is_train:
            self.netG.train()

            # optimizers
            self.setup_optimizer(train_opt)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.l1_loss = nn.L1Loss()


    def setup_optimizer(self, train_opt):
        # 三个optimizer: main, surrogate_network, entropy_model
        # 1. main
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():
            print(k)
            if 'Intra_H265_Surrogate' in k or 'Quantization_H265_Suggrogate' in k or 'Y_TABLE' in k or 'C_TABLE' in k or k.endswith(".quantiles"):
                if self.rank <= 0:
                    logger.warning('Params [{:s}] is not optimized by Optimizer_G.'.format(k))
            else:
                optim_params.append(v)

        self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_G)

        # 3. surrogate network
        print('='*50, "Surrogate network")
        optim_params = []
        for k, v in self.netG.named_parameters():
            if 'Intra_H265_Surrogate' in k or 'Quantization_H265_Suggrogate' in k:
                optim_params.append(v)
                logger.warning(f'Aux Params {k} will be optimized.')
        self.optimizer_surrogate = torch.optim.Adam(optim_params, lr=train_opt['sur_lr'],
                                                weight_decay = train_opt['sur_weight_decay'],
                                                betas=(train_opt['sur_beta1'], train_opt['sur_beta2']))
        self.optimizers.append(self.optimizer_surrogate)


            
    def feed_data(self, data):
        # 输入统一格式为RGB，在这里进行转换
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        if self.color == 'rgb24':
            # 此时的排列顺序是BGR， ref https://feater.top/ffmpeg/introduction-of-rgb-format/
            output_array = np.zeros_like(data['GT'])
            for b in range( data['GT'].shape[0]):
                for d in range( data['GT'].shape[1]):
                    rgb_image =  rearrange(data['GT'][b, d], 'c h w -> h w c')
                    yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2BGR)
                    output_array[b, d] = rearrange(yuv_image, 'h w c -> c h w')
            self.real_H = torch.tensor(output_array).to(self.device)
        elif self.color in ['yuv444p', 'yuv420p']:
            # SNU film测试时会超出显存
            if data['GT'].shape[-1] >= 720:
                data['GT'] = data['GT'][:,:,:,:256,:256]
            output_array = np.zeros_like(data['GT'])
            for b in range( data['GT'].shape[0]):
                for d in range( data['GT'].shape[1]):
                    rgb_image =  rearrange(data['GT'][b, d], 'c h w -> h w c')
                    yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
                    output_array[b, d] = rearrange(yuv_image, 'h w c -> c h w')
            # data augment
            # if self.is_train:
            self.real_H = torch.tensor(output_array).to(self.device)
            if self.real_H.shape[-1] % 16  != 0 or self.real_H.shape[-2] % 16 != 0:
                self.real_H = self.center_crop_to_multiple_of_8(self.real_H, base=16)
            
        else:
            raise Exception('invalid color type.')


    def optimize_parameters(self, inplace_flag=False, current_step=None):
        self.optimizer_G.zero_grad()
        if self.optimizer_surrogate is not None:
            self.optimizer_surrogate.zero_grad()

        b, t, c, h, w = self.real_H.shape
        center = t // 2
        intval = self.gop // 2
        self.input = self.real_H[:, center - intval:center + intval + 1]  # (b,t,c,h,w)
        qp = random.randint(self.opt['network_G']['x265_qp_min'], self.opt['network_G']['x265_qp_max'])
        qp = torch.tensor(float(qp)).expand(b).reshape(-1, 1) / 50.
        init_I_frame = None
        LF, LF_surrogated, target_bpp = self.netG(x=self.input,  pix_fmt=self.color, init_I_frame=init_I_frame, qp=qp, inplace_flag=inplace_flag)
        target_bpp = torch.mean(target_bpp)

        LR_ref = self.input[:,::2]
        l_forw_fit = F.mse_loss(LF, LR_ref) * 10 # b t c h w
        y = self.Quantization(LF_surrogated)
        out_x = self.netG(x=[y, None], rev=True)
        
        l_back_rec = F.l1_loss(out_x, self.input)
        # total loss
        loss = l_forw_fit + l_back_rec   
        loss.backward()
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # optimize surrogate network
        if self.optimizer_surrogate is not None and self.opt['codec'] != 'none' and self.netG.module.Quantization_H265_Suggrogate is not None:
            self.optimizer_surrogate.zero_grad()
            surro_loss = self.netG(self.input, train_surrogate=True)
            surro_loss.backward()
            self.optimizer_surrogate.step()
        else:
            surro_loss = None

        # set log
        if 'train_loss' not in self.log_dict or self.log_dict['train_loss'] is None:
            self.log_dict['train_loss'] = {
                'l_total': [loss if isinstance(loss, int) else loss.item(), ],
                'l_back_rec': [l_back_rec if isinstance(l_back_rec, int) else l_back_rec.item(), ],
                'l_forw_fit': [l_forw_fit if isinstance(l_forw_fit, int) else l_forw_fit.item(), ],
                'bpp': [target_bpp if isinstance(target_bpp, int) else target_bpp.item()],
            }
        else:
            self.log_dict['train_loss']['l_total'].append(loss if isinstance(loss, int) else loss.item())
            self.log_dict['train_loss']['l_back_rec'].append(l_back_rec if isinstance(l_back_rec, int) else l_back_rec.item())
            self.log_dict['train_loss']['l_forw_fit'].append(l_forw_fit if isinstance(l_forw_fit, int) else l_forw_fit.item())
            self.log_dict['train_loss']['bpp'].append(target_bpp if isinstance(target_bpp, int) else target_bpp.item())


    def test_long(self, input, rev=False):
        self.netG.eval()
        if not rev:
            # from thop import profile
            # macs, params = profile(self.netG, inputs=(input, qp, self.color, True))
            # print("params: %.2f"%(params / 1000000))
            LF = self.netG(x=input, qp=None, pix_fmt=self.color, close_codec=True)
            return LF
        else:
            # qp = torch.tensor(float(qp)).expand(input.shape[0]).reshape(-1, 1) / 50.
            out = self.netG(x=[input, None],rev=True)
            return out


    def test(self):
        # self.netG.eval()
        with torch.no_grad():
            if self.opt['scale_type'] == 'CSTVR':
                b, t, c, h, w = self.real_H.shape
                center = t // 2
                intval = self.gop // 2
                self.input = self.real_H[:, max(center - intval, 0):min(center + intval + 1, self.real_H.size(1))]
                input = self.input.float() / 255.
                LF = self.netG(x=input)
                y = self.Quantization(LF)
                fake_H = self.netG(x=[y, None], rev=True)
                self.fake_H = fake_H
                self.vlr = LF
            elif self.opt['scale_type'] == 'CSTVR_w_surrogate_net':
                b, t, c, h, w = self.real_H.shape
                center = t // 2
                intval = self.gop // 2
                self.input = self.real_H[:, max(center - intval, 0):min(center + intval + 1, self.real_H.size(1))]
                input = self.input.float() / 255.
                if self.opt['qp'] is None:
                    qp = 27
                else:
                    qp = self.opt['qp']
                qp = torch.tensor(float(qp)).expand(b).reshape(-1, 1) / 50.

                LF, LF_compressed, bpp = self.netG(x=input,  pix_fmt=self.color, qp=qp)
                y = self.Quantization(LF_compressed)
                out_x = self.netG(x=[y, None], rev=True)
                fake_H = out_x
                self.fake_H = fake_H
                self.vlr = LF
                self.vlr_compressed = LF_compressed
                self.bpp = bpp

        # self.netG.train()

    def empty_cache(self):
        del self.fake_H, self.vlr, self.vlr_compressed
    
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def frozen_params(self, stage):
        pass



