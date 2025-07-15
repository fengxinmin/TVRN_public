import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
try:
    from tqdm import tqdm
    tqdm_open = True
except:
    tqdm_open = False
    pass
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
from einops import rearrange
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
from utils.padder import InputPadder
import torch.nn.functional as F
from collections import OrderedDict
import options.options as option
from models.VRN_model import TVRNCodecModel as Model
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.functions import ycbcr444_to_420
from PIL import Image
from models.VFI_models.EBME.bi_flownet import BiFlowNet
from models.VFI_models.EBME.fusionnet import FusionNet

from models.VFI_models.XVFI.XVFInet import XVFInet

from models.VRN_model import STAAModel

from models.modules.STDR_Net import Net as STDR_Net

import yaml

'''==========import from our code=========='''
sys.path.append('.')
from utils.pytorch_msssim import ssim_matlab
from utils.functions import ycbcr2rgb, rgb2ycbcr

def center_crop(tensor, target_shape):
    """
    Center crop the tensor to the target shape.
    """
    _, *tensor_shape = tensor.shape
    _, *target_shape = target_shape
    
    slices = []
    for dim, (ts, tt) in enumerate(zip(tensor_shape, target_shape)):
        if ts > tt:
            start = (ts - tt) // 2
            slices.append(slice(start, start + tt))
        else:
            slices.append(slice(0, ts))
    
    return tensor[(slice(None), *slices)]


def center_crop(tensor, target_shape):
    """
    Center crop the tensor to the target shape.
    """
    _, *tensor_shape = tensor.shape
    _, *target_shape = target_shape
    
    slices = []
    for dim, (ts, tt) in enumerate(zip(tensor_shape, target_shape)):
        if ts > tt:
            start = (ts - tt) // 2
            slices.append(slice(start, start + tt))
        else:
            slices.append(slice(0, ts))
    
    return tensor[(slice(None), *slices)]



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# VFIformer
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--name', default='test_vfiformer', type=str)
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--net_name', default='VFIformer', type=str, help='')
parser.add_argument('--window_size', default=8, type=int)
parser.add_argument('--module_scale_factor', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--data_root', default='/home/liyinglu/newData/datasets/vfi/SNU-FILM/',type=str)
parser.add_argument('--testset', default='FILM', type=str, help='FILM')
parser.add_argument('--test_level', default='extreme', type=str, help='easy|medium|hard|extreme')
parser.add_argument('--crop_size', default=192, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_augmentation', default=False, type=bool)
parser.add_argument('--resume', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
parser.add_argument('--resume_flownet', default='', type=str)
parser.add_argument('--save_folder', default='./test_results', type=str)
parser.add_argument('--save_result', action='store_true')
# end
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('-path', type=str, required=True)
parser.add_argument('-mode', type=str, default='crf')
parser.add_argument('-model', type=str, default='TVRN')
parser.add_argument('-qp', type=int, default=27)
# vfips
parser.add_argument('--datasets', type=str, default='perceptual_video')
# parser.add_argument('--model', type=str, default='multiscale_v33')
parser.add_argument('--expdir', type=str, default='/data/fengxm/vimeo90k/pretrained_model/VFIPS/exp/eccv_ms_multiscale_v33/', help='exp dir')
parser.add_argument('--depth_ksize', type=int, default=1, help='depth kernel size')
parser.add_argument('--flow', type=str2bool, default=False, help='model use flow or not')
parser.add_argument('--autodata', type=str2bool, default=True, help='model use autodata or not')
# parser.add_argument('--testset', type=str, default='bvivfi', help='test set')
parser.add_argument('--norm', type=str, default='sigmoid', help='normalization function')
parser.add_argument('-staa_opt', type=str, help='Path to option YMAL file.')
# parser.add_argument('--window_size', type=int, default=2, help='window size')
# parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
parser.add_argument('--checkpoints', type=str, default=None)
# parser.add_argument('--gimm_opt', type=str, default='/code/codes/models_GIMM/gimmvfi_r_arb.yaml')

args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

if args.checkpoints is not None:
    opt['path']['pretrain_model_G'] = args.checkpoints
    print('loading weight from ', opt['path']['pretrain_model_G'])
opt['network_G']['entropy_model'] = False

model_name = args.model

'''==========Model setting=========='''
TTA = True
average_metric = True
if model_name == 'TVRN':
    model = Model(opt)
elif model_name == 'STAA':
    staa_opt = option.parse(args.staa_opt, is_train=False)
    model = STAAModel(staa_opt)
elif model_name == 'EMA':
    import models.VFI_models.EMA.config as cfg
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
    from models.VFI_models.EMA.Trainer import Model
    model = Model(-1)
    model.load_model(path = opt['path']['EMA_model'])
    model.eval()
    model.device()
elif model_name == 'SGM':
    import models.VFI_models.SGM.config_SGM as cfg
    from models.VFI_models.SGM.Trainer_x4k import Model
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 4],
        num_key_points=0.5
    )
    model = Model(-1)
    model.load_model(path = opt['path']['SGM_model'])
    model.eval()
    model.device()
elif model_name == 'UPR':
    from models.VFI_models import UPRModelBase
    model = UPRModelBase()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-base.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda()
    # model.to('cuda:1')
elif model_name == 'UPR_l':
    from models.VFI_models import UPRModelLarge
    model = UPRModelLarge()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-large.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda()
elif model_name == 'UPR_L':
    from models.VFI_models import UPRModelLLarge
    model = UPRModelLLarge()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-llarge.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda()
elif model_name == 'IFRNet':
    from models.VFI_models.IFRNet import Model
    model = Model()
    load_path = "/data/fengxm/vimeo90k/pretrained_model/IFRNet/IFRNet_L_Vimeo90K.pth"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda()
elif model_name == 'VFIformer':
    import torch.nn as nn
    from torch.nn.parallel import DataParallel, DistributedDataParallel

    def load_networks(network, resume, strict=True, net_name=None):
        load_path = resume
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device('cpu'))
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if 'optimizer' or 'scheduler' in net_name:
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=strict)
        return network
    from models.VFI_models.VFIFormer.modules import define_G
    # python FILM_test.py --data_root [your SNU-FILM path] 
    # --test_level [easy/medium/hard/extreme] --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth
    args.test_level = 'medium'
    args.net_name = 'VFIformer'
    args.resume = '/data/fengxm/vimeo90k/pretrained_model/VFIformer/net_220.pth'
    args.dist = False
    args.gpu_ids = [0, ]
    args.device = 'cuda'
    model = define_G(args)
    model = load_networks(model, args.resume, net_name=args.net_name)
    down_scale = 2
elif model_name == 'EBME':
    def load_pretrained_state_dict(module, module_name, module_args):
        load_pretrain = module_args.load_pretrain \
                if "load_pretrain" in module_args else True
        if not load_pretrain:
            print("Train %s from random initialization." % module_name)
            return False

        model_file = module_args.model_file \
                if "model_file" in module_args else ""
        if (model_file == "") or (not os.path.exists(model_file)):
            raise ValueError("Please set the correct path for pretrained %s!" % module_name)

        print("Load pretrained model for %s from %s." % (module_name, model_file))
        rand_state_dict = module.state_dict()
        pretrained_state_dict = torch.load(model_file)

        output_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}

        return output_state_dict

    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = 5
    bi_flownet_args.load_pretrain = True
    bi_flownet_args.model_file = '/data/fengxm/vimeo90k/pretrained_model/EBME/ebme/bi-flownet.pkl'

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = False
    fusionnet_args.load_pretrain = True
    fusionnet_args.model_file = '/data/fengxm/vimeo90k/pretrained_model/EBME/ebme/fusionnet.pkl'

    bi_flownet = BiFlowNet(bi_flownet_args).cuda()
    fusionnet = FusionNet(fusionnet_args).cuda()
    bi_flownet.load_state_dict(load_pretrained_state_dict(bi_flownet, "bi_flownet", bi_flownet_args))
    fusionnet.load_state_dict(load_pretrained_state_dict(fusionnet, "fusionnet", fusionnet_args), strict=True)
elif model_name == 'XVFI':
    xvfi_args = argparse.Namespace()
    xvfi_args.S_trn = 1
    xvfi_args.S_tst = 1
    xvfi_args.batch_size = 16
    xvfi_args.checkpoint_dir = './checkpoint_dir'
    xvfi_args.continue_training = False
    xvfi_args.custom_path = './custom_path'
    xvfi_args.dataset = 'Vimeo'
    xvfi_args.epochs = 200
    xvfi_args.exp_num = 1
    xvfi_args.freq_display = 100
    xvfi_args.gpu = 0
    xvfi_args.img_ch = 3
    xvfi_args.init_lr = 0.0001
    xvfi_args.log_dir = './log_dir'
    xvfi_args.loss_type = 'L1'
    xvfi_args.lr_dec_fac = 0.25
    xvfi_args.lr_dec_start = 0
    xvfi_args.lr_milestones = [100, 150, 180]
    xvfi_args.metrics_types = ['PSNR', 'SSIM', 'tOF']
    xvfi_args.model_dir = 'XVFInet_Vimeo_exp1'
    xvfi_args.module_scale_factor = 2
    xvfi_args.multiple = 8
    xvfi_args.need_patch = True
    xvfi_args.net_object = XVFInet  # 注意这里保持类的引用
    xvfi_args.net_type = 'XVFInet'
    xvfi_args.nf = 64
    xvfi_args.num_thrds = 4
    xvfi_args.patch_size = 256
    xvfi_args.phase = 'test_custom'
    xvfi_args.rec_lambda = 1.0
    xvfi_args.save_img_num = 4
    xvfi_args.saving_flow_flag = False
    xvfi_args.test_data_path = '../Datasets/VIC_4K_1000FPS/test'
    xvfi_args.test_img_dir = './test_img_dir'
    xvfi_args.text_dir = './text_dir'
    xvfi_args.train_data_path = '../Datasets/VIC_4K_1000FPS/train'
    xvfi_args.val_data_path = '../Datasets/VIC_4K_1000FPS/val'
    xvfi_args.vimeo_data_path = './vimeo_triplet'
    xvfi_args.weight_decay = 0
    model = XVFInet(xvfi_args)
    pretrained_state_dict = torch.load('/data/fengxm/vimeo90k/pretrained_model/XVFI/XVFInet_Vimeo_exp1_latest.pt', map_location="cuda:0")['state_dict_Model']
    # output_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items() if k.startswith('module.')}
    model.load_state_dict(pretrained_state_dict, strict=True)
    model = model.cuda().eval()
elif model_name in ['GIMM', 'GIMM+VQE']:
    from models.VFI_models.GIMM.models import create_model
    from argparse import Namespace
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return Namespace(**d)
    with open('/code/codes/models_GIMM/gimmvfi_r_arb.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    gimm_config = dict_to_namespace(config_dict)
    gimm_config.arch.fwarp_type = 'linear'
    gimm_config.arch.normalize_weight = True
    model, _ = create_model(gimm_config.arch)
    model = model.cuda()
    load_path = '/data/fengxm/vimeo90k/pretrained_model/GIMM/gimmvfi_r_arb.pt'
    ckpt = torch.load(load_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    # VQE model
    def load_restoration(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.restoration.'):
                load_net_clean[k[len('module.restoration_module.'):]] = v
            elif k.startswith('restoration_module.'):
                load_net_clean[k[len('restoration_module.'):]] = v
        network.load_state_dict(load_net_clean, strict=strict)
    def load_ranker(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.ranker.'):
                load_net_clean[k[len('module.'):]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

elif model_name == 'MoMo':
    from models.VFI_models.momo.synthesis import SynthesisNet
    from models.VFI_models.momo.diffusion.momo import MoMo

    synth_model = SynthesisNet()
    model = MoMo(synth_model=synth_model)
    assert os.path.exists('/data/fengxm/vimeo90k/pretrained_model/momo/model.pth'), 'path to model checkpoints do not exist!'
    ckpt = torch.load('/data/fengxm/vimeo90k/pretrained_model/momo/model.pth', map_location='cpu')
    param_ckpt = ckpt['model']
    model.load_state_dict(param_ckpt)
    del ckpt
    model.cuda()
    model.eval()
elif model_name == 'CVRS':
    from models.VFI_models.cvrs.arch.IMSM import IND_inv3D
    from models.VFI_models.cvrs.utils.options import yaml_load
    inv_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/config.yml')['network_g']['opt']
    model = IND_inv3D(inv_opt).to('cuda')
    inv_weight_p = os.path.join('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)

    time_factor = 2
    scale_factor = 1
    rescale_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/config.yml')
    if time_factor == 2 and scale_factor == 1:
        from models.VFI_models.cvrs.arch.Mynet_arch import RescalerNet
    else:
        from models.VFI_models.cvrs.arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to('cuda')
    weight = torch.load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/model.pth')
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
    # param = get_model_total_params(rescale_model)
    # print(f'param: {param}')
    
elif model_name == 'CVRS_finetuned':
    # 使用所提出的代理网络对CVRS进行finetune
    from models.VFI_models.cvrs import CSTVRModel as Model
    opt['path']['pretrain_model_G'] = '/model/fengxm/VRN/MIMO_VRN/CSTVR_w_surrogate_net/70000_G.pth'
    model = Model(opt)

LFR_gt, LFR_hq, LFR_lq = None, None, None

print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeodataset   Model: {model_name}   TTA: {TTA}')
path = args.path
dirs = os.listdir(path)
level_list = ['sep_testlist.txt',] 
Quantization_H265_Stream = Quantization_H265_Stream(args.qp,-1,None,opt)
for test_file in level_list:
    psnr_list, ssim_list = [], []
    sigma_list = []
    mse_list = []
    psnr_lr_list, ssim_lr_list = [], []
    psnr_inter_list, ssim_inter_list = [], []
    floLPIPS_list, VFIPS_list = [], []
    LPIPS_list, DISTS_list, FID_list = [], [], []
    file_list = []
    bpp_list = []
    psnr_LFR = []
    LFR_lq_psnr_list, LFR_lq_ssim_list, LFR_hq_psnr_list = [], [], []
    with open(os.path.join(path, test_file), "r") as f:
        for line in f:
            line = line.strip()
            file_list.append(line)
    
    for line_id, line in (enumerate(file_list)):

        I0_path = os.path.join(path, 'sequences', line, 'im1.png')
        I1_path = os.path.join(path, 'sequences', line, 'im2.png')
        I2_path = os.path.join(path, 'sequences', line, 'im3.png')
        I3_path = os.path.join(path, 'sequences', line, 'im4.png')
        I4_path = os.path.join(path, 'sequences', line, 'im5.png')
        I5_path = os.path.join(path, 'sequences', line, 'im6.png')
        I6_path = os.path.join(path, 'sequences', line, 'im7.png')

        if model_name == 'TVRN':

            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]
            LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)  # b t c h w  -> b c t h w
            Quantization_H265_Stream.open_writer('cpu',LF.shape[-1],LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,[0,2,1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            outsouts2 = []
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:,[0,2,1]]
            img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]
            out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
            out_x = padder.unpad(out_x)
            out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
            pred = out_x[1::2]
            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda()
            LFR_lq = padder.unpad(LFR_lq)
            LFR_hq = out_x[[0,2,4,6]]
        elif model_name == 'STAA':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')


            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]

            # 0 1 2 3 4 5 6
            LF_processed = []
            for i in range(0, group_1_pad.shape[0]-1, 2):
                LF = model.netG(x=group_1_pad[i:i+3].unsqueeze(0), rev=False)[0]
                LF_processed.append(LF[0])
                group_1_pad[i+2] = LF[-1]
            LF_processed.append(LF[-1])          
            LF_processed = torch.stack(LF_processed, dim=0)  
            LF = padder.unpad(LF_processed)
            
            Quantization_H265_Stream.open_writer('cpu',LF.shape[-1],LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,[0,2,1]].unsqueeze(0), 'b t c h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            outsouts2 = []
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:,[0,2,1]]
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1], v_seg[1:2], v_seg[2:3], v_seg[3:4]
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            # reconstruct HFR video using the inversed order
            for i in range(3):
                recon_3_frames = model.netG(x=[torch.stack([group_list[2-i].cuda(), group_list[3-i].cuda()], dim=1), None], rev=True)
                out_list.append(padder.unpad(recon_3_frames[:,-1]))
                out_list.append(padder.unpad(recon_3_frames[:,-2]))
                pred_list.append(padder.unpad(recon_3_frames[:,-2]))
                group_list[2-i] = recon_3_frames[:,-3]
            out_list.append(padder.unpad(recon_3_frames[:,-3]))
            out_x = torch.cat(out_list[::-1], dim=0)  # b t c h w
            pred = torch.cat(pred_list[::-1], dim=0)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.cat([img1_lq, img2_lq, img3_lq, img4_lq], dim=0).cuda()
            LFR_hq = out_x[[0,2,4,6]]
        elif model_name == 'EMA':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            
            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred.unsqueeze(0))
                pred_list.append(pred.unsqueeze(0))
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = v_seg.cuda()
            LFR_hq = out_x[[0,2,4,6]]
        elif model_name == 'SGM':   
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                pred = model.hr_inference(group_list[i], group_list[i+1], TTA=TTA, down_scale=1, fast_TTA=False).clamp(0.0, 1.0)
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred) 
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)
        elif model_name in ['UPR', 'UPR_l', 'UPR_L']:
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                pred, _, _ = model(group_list[i], group_list[i+1], time_period = 0.5)
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)
        elif model_name in ['IFRNet']:
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = (img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                img1_lq_pad2, img2_lq_pad2 = (group_list[i], group_list[i+1])
                pred = model.inference(img1_lq_pad2, img2_lq_pad2, embt=embt)
                out_list.append((group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append((group_list[-1]))
            out_x = torch.cat(out_list, dim=0)
            pred = torch.cat(pred_list, dim=0)
        elif model_name in ['x265_latency']:
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')


            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            # gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(7)
            out_x = torch.tensor(v_seg).cuda()
        elif model_name == 'VFIformer':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=64)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                down_scale = 0.5
                img1_down = F.interpolate(group_list[i], scale_factor=down_scale, mode="bilinear", align_corners=False)
                img3_down = F.interpolate(group_list[i+1], scale_factor=down_scale, mode="bilinear", align_corners=False)
                b, c, h, w = img3_down.size()
                if h % 64 != 0 or w % 64 != 0:
                    h_new = math.ceil(h / 64) * 64
                    w_new = math.ceil(w / 64) * 64
                    img1_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
                    img3_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
                    img1_new[:, :, :h, :w] = img1_down
                    img3_new[:, :, :h, :w] = img3_down
                    img1_down = img1_new
                    img3_down = img3_new
                flow_down = model.get_flow(img1_down.cuda(), img3_down.cuda())
                if h % 64 != 0 or w % 64 != 0:
                    flow_down = flow_down[:, :, :h, :w]
                flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale
                pred, _,  = model(img1_lq_pad.cuda(), img2_lq_pad.cuda(), flow_pre=flow)
                
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)      
        elif model_name == 'EBME':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=64)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                # pred = model.inference(group_list[i], group_list[i+1], embt=embt, scale_factor=0.8)
                bi_flow = bi_flownet(group_list[i], group_list[i+1])
                pred = fusionnet(group_list[i], group_list[i+1], bi_flow, time_period=0.5)
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)
        elif model_name == 'XVFI':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                pred = model(torch.stack([group_list[i], group_list[i+1]], dim=2), t_value=torch.tensor(0.5).reshape(group_list[i].shape[0], 1).to(group_list[i].device), is_training=False)
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)
        elif model_name in ['GIMM', 'GIMM+VQE']:
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            
            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                # model inference
                I0, I2 = group_list[i], group_list[i+1]
                xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
                    'cuda', non_blocking=True
                )  # b,c,2,h,w
                batch_size = xs.shape[0]
                s_shape = xs.shape[-2:]
                time_step = 2
                coord_inputs = [
                    (
                        model.sample_coord_input(
                            batch_size,
                            s_shape,
                            [(j + 1) * (1.0 / time_step)],
                            device=xs.device,
                        ),
                        None,
                    )
                    for j in range(time_step - 1)
                ]
                t = [
                    (i + 1) * (1.0 / time_step)
                            * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                            for i in range(time_step - 1)]
                with torch.no_grad():
                    all_outputs = model(
                        xs,
                        coord_inputs,
                        t=t,
                    )

                # all_outputs = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
                pred = all_outputs["imgt_pred"][0]
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = v_seg.cuda()
            LFR_hq = out_x[[0,2,4,6]]

        elif model_name == 'MoMo':
            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            
            Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(), v_seg[1:2].cuda(), v_seg[2:3].cuda(), v_seg[3:4].cuda()
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []
            for i in range(3):
                # pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
                pred, _ = model(
                    torch.stack([group_list[i], group_list[i+1]], dim=2).cuda(),
                    num_inference_steps=20,
                    resize_to_fit=True,
                    pad_to_fit_unet=False,
                )
                
                pred = padder.unpad(pred)
                # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = v_seg.cuda()
            LFR_hq = out_x[[0,2,4,6]]

        elif model_name in ['CVRS']:

            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]

            down_size = (4, group_1_pad.shape[-2] , group_1_pad.shape[-1])


            x_down = rescale_model.inference_down(rearrange(ycbcr2rgb(group_1_pad), 't c h w -> c t h w').unsqueeze(0), down_size)
            LR_img = model.inference_latent2RGB(x_down)
            LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> b t c h w')[0]).unsqueeze(0), 'b t c h w -> b c t h w')

            # # debug
            # v_seg = rearrange(LF, 'b c t h w -> b t c h w')[0]
            # LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)  # b t c h w  -> b c t h w
            Quantization_H265_Stream.open_writer('cpu',LF.shape[-1],LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,[0,2,1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            outsouts2 = []
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:,[0,2,1]]
            img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]

            rev_back = model.inference_RGB2latent(rearrange(ycbcr2rgb(v_seg).unsqueeze(0).cuda(), 'b t c h w -> b c t h w'))
            out_x = rescale_model.inference_up(rev_back, (group_1_pad.shape[0], group_1_pad.shape[-2], group_1_pad.shape[-1]))
            # out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
            out_x = padder.unpad(out_x)
            # b c t h w
            out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
            out_x = rgb2ycbcr(out_x)
            pred = out_x[1::2]
            # pred = rgb2ycbcr(pred)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda()
            LFR_lq = padder.unpad(LFR_lq)
            LFR_hq = out_x[[0,2,4,6]]


        elif model_name in ['CVRS_finetuned']:

            img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')
            img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda()), 'c h w -> h w c')

            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]

            down_size = (4, group_1_pad.shape[-2] , group_1_pad.shape[-1])

            LF = model.test_long(input=group_1_pad.unsqueeze(0), rev=False)  # b t c h w  -> b c t h w
            b = LF.shape[0]
            LF = rearrange((rearrange(LF, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b)

            # # compression in yuv444p format
            Quantization_H265_Stream.open_writer('cpu',LF.shape[-1],LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,[0,2,1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _,img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            outsouts2 = []
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:,[0,2,1]]
            img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]
            out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(), 'b t c h w -> b c t h w'), rev=True)
            out_x = padder.unpad(out_x)
            # b c t h w
            out_x = rearrange(out_x, 'b t c h w -> b t c h w')[0]
            # out_x = rgb2ycbcr(out_x)
            pred = out_x[1::2]
            # pred = rgb2ycbcr(pred)

            LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda()
            LFR_lq = padder.unpad(LFR_lq)
            LFR_hq = out_x[[0,2,4,6]]
        else:
            raise Exception('invalid model name')
            
        if model_name == 'x265_latency':
            out_rgb = ycbcr2rgb(out_x)
            out_label_rgb = ycbcr2rgb(out_label)
            ssim = ssim_matlab(out_label_rgb, torch.round(out_rgb * 255) / 255.).detach().cpu().numpy()
            out_label_rgb = out_label_rgb.cuda()      
            out_rgb = out_rgb.cuda()
            out_x = (np.round(out_x.cpu().numpy() * 255) / 255.).clip(min=0, max=1)
            out_label = out_label.cpu().numpy()
            psnr = -10 * math.log10(((out_label - out_x) * (out_label - out_x)).mean())
            mse = ((out_label - out_x) * (out_label - out_x)).mean()
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse)
            bpp_list.append(img_distri)
            continue


        out_rgb = ycbcr2rgb(out_x)
        out_label_rgb = ycbcr2rgb(out_label)
        gt_rgb = ycbcr2rgb(gt)
        pred_rgb = ycbcr2rgb(pred)
        ssim = ssim_matlab(out_label_rgb, torch.round(out_rgb * 255) / 255.).detach().cpu().numpy()
        ssim_inter = ssim_matlab(gt_rgb, torch.round(pred_rgb * 255) / 255.).detach().cpu().numpy()

        out_label_rgb = out_label_rgb.cuda()      
        out_rgb = out_rgb.cuda()
        gt_rgb = gt_rgb.cuda()
        pred_rgb = pred_rgb.cuda()

        FID_list.append(0)
        
        out_x = (np.round(out_x.cpu().numpy() * 255) / 255.).clip(min=0, max=1)
        out_label = out_label.cpu().numpy()
        # average mse
        psnr = -10 * math.log10(((out_label - out_x) * (out_label - out_x)).mean())

        mse = ((out_label - out_x) * (out_label - out_x)).mean()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)
        sigma_list.append(np.std([-10 * math.log10(((out_label[i] - out_x[i]) * (out_label[i] - out_x[i])).mean()) for i in range(out_label.shape[0])]))

        psnr_inter = -10 * math.log10(((gt.cpu().numpy() - pred.cpu().numpy()) * (gt.cpu().numpy() - pred.cpu().numpy())).mean())
        psnr_inter_list.append(psnr_inter)
        ssim_inter_list.append(ssim_inter)

        if LFR_gt is not None:
            LFR_lq_psnr_list.append(-10 * math.log10(((LFR_gt.cpu().numpy() - LFR_lq.cpu().numpy()) * (LFR_gt.cpu().numpy() - LFR_lq.cpu().numpy())).mean()))
            LFR_hq_psnr_list.append(-10 * math.log10(((LFR_gt.cpu().numpy() - LFR_hq.cpu().numpy()) * (LFR_gt.cpu().numpy() - LFR_hq.cpu().numpy())).mean()))
            LFR_lq_ssim_list.append(ssim_matlab(ycbcr2rgb(LFR_gt), torch.round(ycbcr2rgb(LFR_lq) * 255) / 255.).detach().cpu().numpy())
        else:
            LFR_lq_psnr_list.append(0)
            LFR_hq_psnr_list.append(0)

        print(img_distri)

        # # 可视化
        if test_file == 'sep_testlist_viz.txt':
            import matplotlib.pyplot as plt
            fig_name = line.replace('/', '_')
            if not os.path.exists(f'/output/{fig_name}'):
                os.makedirs(f'/output/{fig_name}/HFR_recon/')
                os.makedirs(f'/output/{fig_name}/LFR_lq/')
                os.makedirs(f'/output/{fig_name}/LFR_hq/')
            if LFR_gt is not None:
                for frm_id in range(LFR_gt.shape[0]):
                    plt.imsave(f'/output/{fig_name}/LFR_lq/frame{frm_id}_qp{args.qp}_{model_name}_psnr{-10 * math.log10(((LFR_gt[frm_id].cpu().numpy() - LFR_lq[frm_id].cpu().numpy()) * (LFR_gt[frm_id].cpu().numpy() - LFR_lq[frm_id].cpu().numpy())).mean()):.2f}_bpp{img_distri:.4f}.png', rearrange(ycbcr2rgb(LFR_lq), 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0,1))
                    plt.imsave(f'/output/{fig_name}/LFR_hq/frame{frm_id}_qp{args.qp}_{model_name}_psnr{-10 * math.log10(((LFR_gt[frm_id].cpu().numpy() - LFR_hq[frm_id].cpu().numpy()) * (LFR_gt[frm_id].cpu().numpy() - LFR_hq[frm_id].cpu().numpy())).mean()):.2f}_bpp{img_distri:.4f}.png', rearrange(ycbcr2rgb(LFR_hq), 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0,1))

            for frm_id in range(out_label.shape[0]):
                psnr_viz = -10 * math.log10(((out_label[frm_id] - out_x[frm_id]) * (out_label[frm_id] - out_x[frm_id])).mean())
                ssim_viz = ssim_matlab(out_label_rgb[frm_id:frm_id+1], torch.round(out_rgb[frm_id:frm_id+1] * 255) / 255.).detach().cpu().numpy()
                plt.imsave(f'/output/{fig_name}/HFR_recon/frame{frm_id}_qp{args.qp}_{model_name}_psnr{psnr_viz:.2f}_ssim{ssim_viz:.4f}_bpp{img_distri:.4f}.png', rearrange(out_rgb, 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0,1))
                plt.imsave(f'/output/{fig_name}/HFR_recon/frame{frm_id}_gt.png', rearrange(out_label_rgb, 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0,1))
        bpp_list.append(img_distri)
        pass
        
    print("Vimeo dataset")
    print("QP: ", args.qp)
    print(f"Model: {model_name}, test file: {test_file}")
    if model_name == 'x265_latency':
        print(f"psnr:{np.mean(psnr_list):.6f},psnr_avg_mse:{ -10 * math.log10(np.mean(mse_list)):.6f},ssim:{np.mean(ssim_list):.6f},psnr_LFR_lq:{0},psnr_LFR_hq:{0},psnr inter:{0},ssim inter:{0},lpips:{0},fid:{0},dists:{0},ave_img_bpp:{np.mean(bpp_list):.6f}")
    else:
        print(f"psnr:{np.mean(psnr_list):.6f},psnr_avg_mse:{ -10 * math.log10(np.mean(mse_list)):.6f},ssim:{np.mean(ssim_list):.6f},psnr_LFR_lq:{np.mean(LFR_lq_psnr_list):.6f},ssim_LFR_lq:{np.mean(LFR_lq_ssim_list):.6f},psnr_LFR_hq:{np.mean(LFR_hq_psnr_list):.6f},sigma:{np.mean(sigma_list)},psnr inter:{np.mean(psnr_inter_list):.6f},ssim inter:{np.mean(ssim_inter_list):.6f},lpips:{np.mean(LPIPS_list):.6f},fid:{np.mean(FID_list):.6f},dists:{np.mean(DISTS_list):.6f},ave_img_bpp:{np.mean(bpp_list):.6f}")

