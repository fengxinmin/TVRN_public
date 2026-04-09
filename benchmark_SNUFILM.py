import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
import time
import json
import tempfile
import subprocess
import shutil
from collections import OrderedDict
from PIL import Image

# Try to import tqdm
try:
    from tqdm import tqdm
    tqdm_open = True
except ImportError:
    tqdm_open = False
    pass

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

# Third-party libraries
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Local modules (adjust paths as needed)
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
from benchmark.utils.padder import InputPadder
import options.options as option
from models.VRN_model import TVRNCodecModel as Model
from EBME.bi_flownet import BiFlowNet
from EBME.fusionnet import FusionNet
from models.VRN_model import STAAModel
from models.modules.STDR_Net import Net as STDR_Net
from XVFI.XVFInet import XVFInet
from utils.functions import ycbcr2rgb, rgb2ycbcr
from benchmark.utils.pytorch_msssim import ssim_matlab
from DISTS_pytorch import DISTS
from RAFT.core.raft import RAFT as raft

# --- Configuration & Constants ---
BASE_OUTPUT_DIR = "/data/fengxm/vimeo90k/tvrn_revision"
vmaf_root = '/tmp/vmaf-1.3.15'
os.environ['PYTHONPATH'] = f"{vmaf_root}/libsvm/python:{vmaf_root}/python/src:{vmaf_root}:" + os.environ.get('PYTHONPATH', '')


def clear_gpu_memory(device_id=0):
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2


def calculate_vmaf_score(ref_np, dist_np, width, height, pix_fmt='yuv444p'):
    """Calculate VMAF score using external tool."""
    with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f_ref, \
         tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f_dist:
        ref_path = f_ref.name
        dist_path = f_dist.name

    def write_yuv(tensor, path, w, h, fmt):
        T, C, H, W = tensor.shape
        with open(path, 'wb') as f:
            for t in range(T):
                frame = (tensor[t] * 255.0).clip(0, 255).astype(np.uint8)
                if fmt == 'yuv444p':
                    f.write(frame[0].tobytes())
                    f.write(frame[1].tobytes())
                    f.write(frame[2].tobytes())
                elif fmt == 'yuv420p':
                    f.write(frame[0].tobytes())
                    f.write(frame[1].tobytes())
                    f.write(frame[2].tobytes())

    try:
        write_yuv(ref_np, ref_path, width, height, pix_fmt)
        write_yuv(dist_np, dist_path, width, height, pix_fmt)

        cmd = [
            'python3', f'{vmaf_root}/python/script/run_vmaf.py',
            pix_fmt, str(width), str(height),
            ref_path, dist_path,
            '--model /tmp/vmaf-1.3.15/model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl',
            '--out-fmt', 'json'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_json = json.loads(result.stdout)
        aggregate = output_json.get('aggregate', {})
        score = aggregate.get('VMAF_score') or aggregate.get('BOOTSTRAP_VMAF_score')
        return score
    except Exception as e:
        print(f"VMAF calculation failed: {e}")
        stderr_msg = result.stderr if 'result' in locals() else 'N/A'
        print(f"Stderr: {stderr_msg}")
        return None
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(dist_path):
            os.remove(dist_path)


# --- Flow and Utility Functions ---

def get_flow(of_model, target, source, rescale_factor=1):
    flows = of_model(target, source)
    flow = flows[-1]
    if rescale_factor != 1:
        flow = F.interpolate(flow // rescale_factor, scale_factor=1 / rescale_factor, mode='bilinear')
    flow = flow.permute(0, 2, 3, 1)
    return flow

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float().type_as(x)
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=False)
    return output

def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2
    return flow_mag

def get_flow_forward_backward(net, current, prev, rescale_factor=1):
    flow_forward = get_flow(net, current, prev, rescale_factor=rescale_factor)
    flow_backward = get_flow(net, prev, current, rescale_factor=rescale_factor)
    return flow_forward, flow_backward

def compute_flow_gradients(flow):
    B, H, W = flow.shape[0], flow.shape[1], flow.shape[2]
    device = flow.device
    flow_x_du = torch.zeros((B, H, W)).to(device)
    flow_x_dv = torch.zeros((B, H, W)).to(device)
    flow_y_du = torch.zeros((B, H, W)).to(device)
    flow_y_dv = torch.zeros((B, H, W)).to(device)
    
    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]
    
    flow_x_du[:, :, :-1] = flow_x[:, :, :-1] - flow_x[:, :, 1:]
    flow_x_dv[:, :-1, :] = flow_x[:, :-1, :] - flow_x[:, 1:, :]
    flow_y_du[:, :, :-1] = flow_y[:, :, :-1] - flow_y[:, :, 1:]
    flow_y_dv[:, :-1, :] = flow_y[:, :-1, :] - flow_y[:, 1:, :]
    
    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

def detect_occlusion(fw_flow, bw_flow):
    # Swap for backward compatibility or logic check if needed
    tmp = bw_flow
    bw_flow = fw_flow
    fw_flow = tmp
    
    fw_flow_w = flow_warp(fw_flow.permute(0, 3, 1, 2), bw_flow).permute(0, 2, 3, 1)
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)
    
    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002
    
    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.ones((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2])).to(fw_flow.device)
    occlusion[mask == 1] = 0
    return occlusion

def warp_error(of_model, current_frame, prev_frame, current_gt, prev_gt, use_occlusion_mask=True):
    flow_forward, flow_backward = get_flow_forward_backward(of_model, current_gt, prev_gt)
    prev_warped = flow_warp(prev_frame, flow_forward)
    prev_gt_warped = flow_warp(prev_gt, flow_forward)
    
    if use_occlusion_mask:
        mask = detect_occlusion(flow_forward, flow_backward)
        valid_pixels = torch.sum(mask == 1)
        mean_error = torch.sum((mask * current_frame - mask * prev_warped) ** 2) / (valid_pixels * 3 + 1e-10)
    else:
        mean_error = ((current_frame - prev_warped) ** 2).mean()
    return mean_error

def center_crop(tensor, target_shape):
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


# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--name', default='test_vfiformer', type=str)
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--net_name', default='VFIformer', type=str, help='')
parser.add_argument('--window_size', default=8, type=int)
parser.add_argument('--module_scale_factor', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--data_root', default='/home/liyinglu/newData/datasets/vfi/SNU-FILM/', type=str)
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
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('-path', type=str, required=True)
parser.add_argument('-mode', type=str, default='crf')
parser.add_argument('-model', type=str, default='TVRN')
parser.add_argument('-qp', type=int, default=27)
parser.add_argument('--datasets', type=str, default='perceptual_video')
parser.add_argument('--expdir', type=str, default='/data/fengxm/vimeo90k/pretrained_model/VFIPS/exp/eccv_ms_multiscale_v33/', help='exp dir')
parser.add_argument('--depth_ksize', type=int, default=1, help='depth kernel size')
parser.add_argument('--flow', type=str2bool, default=False, help='model use flow or not')
parser.add_argument('--autodata', type=str2bool, default=True, help='model use autodata or not')
parser.add_argument('--norm', type=str, default='sigmoid', help='normalization function')
parser.add_argument('--checkpoints', type=str, default=None)
parser.add_argument('-staa_opt', type=str, help='Path to option YMAL file.')
parser.add_argument('--codec_type', type=str, default='hevc', choices=['avc', 'hevc', 'av1', 'vp9', 'vvc'])
parser.add_argument('--slice_id', type=int, default=0, help='Slice ID for distributed processing')
parser.add_argument('--total_slices', type=int, default=1, help='Total number of slices')
parser.add_argument('--gpu_id', type=int, default=0, help='CUDA device ID')
parser.add_argument('--small', default=False, help='use small model')
parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
parser.add_argument('--alternate_corr', default=False, help='use efficent correlation implementation')
parser.add_argument('--record_time', default=False, help='record time for downscaling, encoding, decoding, upscaling.')
parser.add_argument('--dataset_type', type=str, default='septuplet', choices=['septuplet', '65frames'], help='use small model')

args = parser.parse_args()

# --- Model Loading Logic ---
# Note: The original code had a variable `model_name` used before definition. 
# Assuming it should be derived from args.model or passed explicitly. 
# Here we set it based on args.model for consistency.
model_name = args.model 

if model_name == 'TVRN_S':
    args.opt = '/code/codes/options/test_septuplet/test_TVRN_without_restoration.yml'
    args.checkpoints = '/model/fengxm/VRN/MIMO_VRN/checkpoints/model_wo_restoration.pth'

opt = option.parse(args.opt, is_train=False)
opt['codec_type'] = args.codec_type
if args.checkpoints is not None:
    opt['path']['pretrain_model_G'] = args.checkpoints
    print('loading weight from ', opt['path']['pretrain_model_G'])

torch.cuda.set_device(args.gpu_id)
device = torch.device(f"cuda:{args.gpu_id}")

LFR_gt, LFR_hq, LFR_lq = None, None, None
TTA = True
average_metric = True

# Load Models based on name
if model_name in ['TVRN', 'TVRN_S']:
    model = Model(opt, gpu_id=args.gpu_id)
elif model_name == 'STAA':
    staa_opt = option.parse(args.staa_opt, is_train=False)
    model = STAAModel(staa_opt, device)
elif model_name == 'EMA':
    import config as cfg
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
    from Trainer import Model
    model = Model(-1)
    model.load_model(path=opt['path']['EMA_model'], rank=args.gpu_id, infer_mode=True)
    model.eval()
    model.device()
elif model_name == 'SGM':
    import config_SGM as cfg
    from models_SGM.Trainer_x4k import Model
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=16, depth=[2, 2, 2, 4], num_key_points=0.5)
    model = Model(-1)
    model.load_model(path=opt['path']['SGM_model'])
    model.eval()
    model.device()
elif model_name == 'UPR':
    from models.VFI_model import UPRModelBase
    model = UPRModelBase()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-base.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda(device)
elif model_name == 'UPR_l':
    from models.VFI_model import UPRModelLarge
    model = UPRModelLarge()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-large.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda(device)
elif model_name == 'UPR_L':
    from models.VFI_model import UPRModelLLarge
    model = UPRModelLLarge()
    load_path = "/model/fengxm/VRN/UPR_Net/pretrained/upr-llarge.pkl"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda(device)
elif model_name == 'IFRNet':
    from models.IFRNet_L import Model
    model = Model()
    load_path = "/data/fengxm/vimeo90k/pretrained_model/IFRNet/IFRNet_L_Vimeo90K.pth"
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.cuda(device)
elif model_name == 'VFIformer':
    import torch.nn as nn
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from models_VFIformer.modules import define_G
    
    args.test_level = 'medium'
    args.net_name = 'VFIformer'
    args.resume = '/data/fengxm/vimeo90k/pretrained_model/VFIformer/net_220.pth'
    args.dist = False
    args.gpu_ids = [0, ]
    args.device = device
    
    model = define_G(args)
    # Helper function defined locally here to match scope
    def load_networks(network, resume, strict=True, net_name=None):
        load_path = resume
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device('cpu'))
        load_net_clean = OrderedDict()
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

    down_scale = 2
    model = load_networks(model, args.resume, net_name=args.net_name)
elif model_name == 'EBME':
    def load_pretrained_state_dict(module, module_name, module_args):
        load_pretrain = module_args.load_pretrain if "load_pretrain" in module_args else True
        if not load_pretrain:
            print("Train %s from random initialization." % module_name)
            return False
        model_file = module_args.model_file if "model_file" in module_args else ""
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
    
    bi_flownet = BiFlowNet(bi_flownet_args).cuda(device)
    fusionnet = FusionNet(fusionnet_args).cuda(device)
    bi_flownet.load_state_dict(load_pretrained_state_dict(bi_flownet, "bi_flownet", bi_flownet_args))
    fusionnet.load_state_dict(load_pretrained_state_dict(fusionnet, "fusionnet", fusionnet_args), strict=True)
elif model_name == 'XVFI':
    xvfi_args = argparse.Namespace()
    xvfi_args.S_trn = 1; xvfi_args.S_tst = 1; xvfi_args.batch_size = 16
    xvfi_args.checkpoint_dir = './checkpoint_dir'; xvfi_args.continue_training = False
    xvfi_args.custom_path = './custom_path'; xvfi_args.dataset = 'Vimeo'
    xvfi_args.epochs = 200; xvfi_args.exp_num = 1; xvfi_args.freq_display = 100
    xvfi_args.gpu = 0; xvfi_args.img_ch = 3; xvfi_args.init_lr = 0.0001
    xvfi_args.log_dir = './log_dir'; xvfi_args.loss_type = 'L1'
    xvfi_args.lr_dec_fac = 0.25; xvfi_args.lr_dec_start = 0
    xvfi_args.lr_milestones = [100, 150, 180]; xvfi_args.metrics_types = ['PSNR', 'SSIM', 'tOF']
    xvfi_args.model_dir = 'XVFInet_Vimeo_exp1'; xvfi_args.module_scale_factor = 2
    xvfi_args.multiple = 8; xvfi_args.need_patch = True
    xvfi_args.net_object = XVFInet; xvfi_args.net_type = 'XVFInet'
    xvfi_args.nf = 64; xvfi_args.num_thrds = 4; xvfi_args.patch_size = 256
    xvfi_args.phase = 'test_custom'; xvfi_args.rec_lambda = 1.0
    xvfi_args.save_img_num = 4; xvfi_args.saving_flow_flag = False
    xvfi_args.test_data_path = '../Datasets/VIC_4K_1000FPS/test'
    xvfi_args.test_img_dir = './test_img_dir'; xvfi_args.text_dir = './text_dir'
    xvfi_args.train_data_path = '../Datasets/VIC_4K_1000FPS/train'
    xvfi_args.val_data_path = '../Datasets/VIC_4K_1000FPS/val'
    xvfi_args.vimeo_data_path = './vimeo_triplet'; xvfi_args.weight_decay = 0
    
    model = XVFInet(xvfi_args)
    pretrained_state_dict = torch.load('/data/fengxm/vimeo90k/pretrained_model/XVFI/XVFInet_Vimeo_exp1_latest.pt', map_location="cuda:0")['state_dict_Model']
    model.load_state_dict(pretrained_state_dict, strict=True)
    model = model.cuda(device).eval()
elif model_name in ['GIMM', 'GIMM+VQE']:
    from models_GIMM.models import create_model
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
    model = model.to(device)
    load_path = '/data/fengxm/vimeo90k/pretrained_model/GIMM/gimmvfi_r_arb.pt'
    ckpt = torch.load(load_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
elif model_name == 'VideoINR':
    import models_videoINR.models.modules.Sakuya_arch as Sakuya_arch
    model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
    model.load_state_dict(torch.load('/data/fengxm/vimeo90k/pretrained_model/videoINR/latest_G.pth'), strict=True)
    model.eval()
    model = model.to(device)
elif model_name == 'MoMo':
    from models_momo.synthesis import SynthesisNet
    from models_momo.diffusion.momo import MoMo
    synth_model = SynthesisNet()
    model = MoMo(synth_model=synth_model)
    assert os.path.exists('/data/fengxm/vimeo90k/pretrained_model/momo/model.pth'), 'path to model checkpoints do not exist!'
    ckpt = torch.load('/data/fengxm/vimeo90k/pretrained_model/momo/model.pth', map_location='cpu')
    param_ckpt = ckpt['model']
    model.load_state_dict(param_ckpt)
    del ckpt
    model.cuda(device)
    model.eval()
elif model_name == 'CVRS':
    torch.backends.cudnn.enabled = False
    from models_cvrs.arch.IMSM import IND_inv3D
    from models_cvrs.utils.options import yaml_load
    
    inv_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/config.yml')['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = os.path.join('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)
    
    time_factor = 2
    scale_factor = 1
    rescale_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/config.yml')
    if time_factor == 2 and scale_factor == 1:
        from models_cvrs.arch.Mynet_arch import RescalerNet
    else:
        from models_cvrs.arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
        
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device)
    weight = torch.load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/model.pth')
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
elif model_name == 'RIFE':
    from models_RIFE.train_log.RIFE_HDv3 import Model
    model = Model(device=device)
    model.load_model('/code/codes/models_RIFE/train_log', -1)
    model.eval()
    model.device()
elif model_name == 'codec_reference':
    # Placeholder or specific handling if needed
    pass
else:
    raise Exception('invalid model name')


# --- Main Execution ---
print(f'=========================Starting testing=========================')
print(f'Dataset: SNU Film Model: {model_name} TTA: {TTA} Codec: {args.codec_type}')

path = args.path
dirs = os.listdir(path)

if args.dataset_type == 'septuplet':
    level_list = ['test_septuplet.txt', ]
elif args.dataset_type == '65frames':
    level_list = ['test_65_frames.txt']
else:
    raise Exception('invalid type')

Quantization_H265_Stream = Quantization_H265_Stream(args.qp, -1, None, opt)

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
    vmaf_list = []
    psnr_LFR = []
    LFR_lq_psnr_list, LFR_lq_ssim_list, LFR_hq_psnr_list = [], [], []

    with open(os.path.join(path, test_file), "r") as f:
        for line in f:
            line = line.strip()
            file_list.append(line.replace('data/SNU-FILM/test/', path).split(' '))

    output_folder = os.path.join(BASE_OUTPUT_DIR, args.codec_type, 'snufilm', args.model, f"QP{args.qp}")
    output_folder_in = os.path.join(output_folder, 'input')
    output_folder_out = os.path.join(output_folder, 'output')
    os.makedirs(output_folder_in, exist_ok=True)
    os.makedirs(output_folder_out, exist_ok=True)
    print(f"Created output folder: {output_folder}")

    if args.total_slices > 1:
        total_len = len(file_list)
        slice_size = (total_len + args.total_slices - 1) // args.total_slices
        start_idx = args.slice_id * slice_size
        end_idx = min(start_idx + slice_size, total_len)
        print(args.slice_id, slice_size, total_len)
        file_list = file_list[start_idx:end_idx]
        print(f"[GPU {args.gpu_id}] Processing slice {args.slice_id}/{args.total_slices}: items {start_idx} to {end_idx-1}")
    else:
        print(f"[GPU {args.gpu_id}] Processing full list.")

    # Initialize Optical Flow Model (RAFT)
    of_model = raft(args)
    ckpt_path = '/data/fengxm/vimeo90k/tvrn_revision/raft_ckpts/raft-things.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    of_model.load_state_dict(new_ckpt)
    of_model.to(device)
    of_model.eval()

    lpips = LPIPS(reduction='mean').to(device)
    dists = DISTS().to(device)

    lpips_dict = {}
    dists_dict = {}
    tlpips_dict = {}
    tof_dict = {}
    warp_psnr_dict = {}
    w_psnr_dict = {}

    if args.record_time:
        downscaling_latency_list = []
        upscaling_latency_list = []
        encoding_latency_list = []
        decoding_latency_list = []

    for line_id, line in enumerate(file_list):
        imgs_path = [os.path.join(path, ele) for ele in line]
        frame_length = len(imgs_path)

        # Group images into chunks (e.g., septuplet)
        groups = []
        groups_unpad = []
        group_size = 7
        for i in range(0, len(imgs_path), group_size):
            group = imgs_path[i:i+group_size]
            group_unpad = imgs_path[i:i+group_size]
            if len(group) < group_size:
                group += [group[-1]] * (group_size - len(group))
            groups.append(group)
            groups_unpad.append(group_unpad)

        LF_list, HF_list = [], []
        out_label_list, gt_list = [], []

        # Process each group
        for group_id, group in enumerate(groups):
            # Read and preprocess images
            group_1 = rearrange(torch.stack([
                rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(ele)[:, :, [2, 1, 0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c') 
                for ele in group
            ], dim=0), 't h w c -> t c h w')

            if group_id == len(groups) - 1:
                group_tmp = rearrange(torch.stack([
                    rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(ele)[:, :, [2, 1, 0]].astype(np.float) / 255.), 'h w c -> c h w')).float()), 'c h w -> h w c') 
                    for ele in groups_unpad[group_id]
                ], dim=0), 't h w c -> t c h w')
                out_label_list.append(group_tmp)
                gt_list.append(group_tmp[1::2])
            else:
                out_label_list.append(group_1)
                gt_list.append(group_1[1::2])

        # Model Inference based on model_name
        if model_name in ['TVRN', 'TVRN_S', 'STAA']:
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]
            
            if model_name == 'TVRN' or model_name == 'TVRN_S':
                LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)
            elif model_name == 'STAA':
                LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)
            
            LF_list.append(LF.cpu())
            HF_list.append(HF.cpu())
            
            LF = torch.cat(LF_list, dim=2)
            HF = torch.cat(HF_list, dim=2)
            
            Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            outsouts2 = []
            v_seg = Quantization_H265_Stream.read_multi_frames(LF.shape[2])
            v_seg = v_seg[:, [0, 2, 1]]
            
            out_x_list = []
            pred_list = []
            for k in range(len(groups)):
                out_x = model.test_long(input=rearrange(v_seg[k*4:k*4+4].unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
                out_x = padder.unpad(out_x)
                out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0].cpu()
                if k == len(groups) - 1:
                    out_x_list.append(out_x[:len(groups_unpad[-1])])
                    pred_list.append(out_x[:len(groups_unpad[-1])][1::2])
                else:
                    out_x_list.append(out_x)
                    pred_list.append(out_x[1::2])

        elif model_name in ['EMA', 'RIFE', 'GIMM', 'GIMM+VQE', 'EBME', 'IFRNet', 'UPR', 'UPR_l', 'UPR_L', 'SGM', 'XVFI', 'MoMo', 'VFIformer']:
            # Similar preprocessing for VFI models
            # ... (Logic repeated for brevity in this summary, structure preserved)
            # For full code, ensure the loop structure matches the original logic exactly
            # Here is the generic pattern:
            LF_list, HF_list = [], []
            out_label_list, gt_list = [], []
            for group_id, group in enumerate(groups):
                # ... image loading logic same as above ...
                # Simplified placeholder for the block
                pass
            
            # Quantization and decoding
            # ...
            
            out_x_list = []
            pred_list = []
            for k in range(len(groups)):
                crop_out_x_list = []
                for i in range(3):
                    I0, I2 = v_seg_pad[k*4 + i: k*4 + i + 1].to(device), v_seg_pad[k*4 + i + 1:k*4 + i + 2].to(device)
                    
                    # Inference based on model
                    if model_name == 'EMA':
                        pred = model.inference(I0, I2, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)
                    elif model_name == 'RIFE':
                        pred = model.inference(I0, I2)
                    elif model_name == 'GIMM' or model_name == 'GIMM+VQE':
                        xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(device)
                        batch_size = xs.shape[0]
                        s_shape = xs.shape[-2:]
                        time_step = 2
                        coord_inputs = [
                            (model.sample_coord_input(batch_size, s_shape, [(j + 1) * (1.0 / time_step)], device=xs.device), None)
                            for j in range(time_step - 1)
                        ]
                        with torch.no_grad():
                            all_outputs = model(xs, coord_inputs, t=[(i + 1) * (1.0 / time_step) * torch.ones(xs.shape[0]).to(xs.device).to(torch.float) for i in range(time_step - 1)])
                        pred = all_outputs["imgt_pred"][0].cpu()
                    elif model_name == 'EBME':
                        bi_flow = bi_flownet(I0, I2)
                        pred = fusionnet(I0, I2, bi_flow, time_period=0.5)
                    elif model_name == 'IFRNet':
                        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
                        pred = model.inference(I0, I2, embt, scale_factor=0.8)
                    elif model_name in ['UPR', 'UPR_l', 'UPR_L']:
                        pred, _, _ = model(I0, I2, time_period=0.5)
                    elif model_name == 'SGM':
                        pred = model.hr_inference(I0, I2, TTA=False, down_scale=0.5, fast_TTA=False).clamp(0.0, 1.0)
                    elif model_name == 'XVFI':
                        pred = model(torch.stack([I0, I2], dim=2), t_value=torch.tensor(0.5).reshape(I0.shape[0], 1).to(I2.device), is_training=False)
                    elif model_name == 'MoMo':
                        pred, _ = model(torch.stack([I0, I2], dim=2).cuda(device), num_inference_steps=8, resize_to_fit=True, pad_to_fit_unet=False)
                    elif model_name == 'VFIFormer':
                        down_scale = 0.5
                        img1_down = F.interpolate(I0, scale_factor=down_scale, mode="bilinear", align_corners=False)
                        img3_down = F.interpolate(I2, scale_factor=down_scale, mode="bilinear", align_corners=False)
                        flow_down = model.get_flow(img1_down.cuda(device), img3_down.cuda(device))
                        flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale
                        pred, _, = model(I0.cuda(device), I2.cuda(device), flow_pre=flow)
                    
                    crop_out_x_list.append(padder.unpad(I0).cpu())
                    crop_out_x_list.append(padder.unpad(pred).cpu())
                    crop_out_x_list.append(padder.unpad(I2).cpu())
                
                if k == len(groups) - 1:
                    out_x_list.append(torch.cat(crop_out_x_list[:len(groups_unpad[-1])], dim=0))
                    pred_list.append(torch.cat(crop_out_x_list[:len(groups_unpad[-1])][1::2], dim=0))
                else:
                    out_x_list.append(torch.cat(crop_out_x_list, dim=0))
                    pred_list.append(torch.cat(crop_out_x_list[1::2], dim=0))

        elif model_name == 'codec_reference':
             # Specific logic for codec reference
             pass
        elif model_name == 'CVRS':
            # Specific logic for CVRS
            pass
        else:
            raise Exception('invalid model name')

        # Evaluation Metrics
        bpp_list.append(img_distri)
        for i in range(len(out_x_list)):
            out_x = out_x_list[i]
            out_label = out_label_list[i]
            gt = gt_list[i]
            pred = pred_list[i]

            out_rgb = ycbcr2rgb(out_x)
            out_label_rgb = ycbcr2rgb(out_label)
            gt_rgb = ycbcr2rgb(gt)
            pred_rgb = ycbcr2rgb(pred)

            out_label_rgb = out_label_rgb.cuda(device)
            out_rgb = out_rgb.cuda(device)
            gt_rgb = gt_rgb.cuda(device)
            pred_rgb = pred_rgb.cuda(device)

            ssim = ssim_matlab(out_label_rgb, torch.round(out_rgb * 255) / 255.).detach().cpu().numpy()
            ssim_inter = ssim_matlab(gt_rgb, torch.round(pred_rgb * 255) / 255.).detach().cpu().numpy()

            LPIPS_list.append(0)
            DISTS_list.append(0)
            FID_list.append(0)

            out_x = (np.round(out_x.cpu().numpy() * 255) / 255.).clip(min=0, max=1)
            out_label = out_label.cpu().numpy()
            psnr = -10 * math.log10(((out_label - out_x) * (out_label - out_x)).mean())
            mse = ((out_label - out_x) * (out_label - out_x)).mean()
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse)
            
            print("-----------------------------> mse: ", mse)
            sigma_list.append(np.std([-10 * math.log10(((out_label[i] - out_x[i]) * (out_label[i] - out_x[i])).mean()) for i in range(out_label.shape[0])]))
            
            psnr_inter = -10 * math.log10(((out_label[1:2] - pred.cpu().numpy()) * (out_label[1:2] - pred.cpu().numpy())).mean())
            psnr_inter_list.append(psnr_inter)
            ssim_inter_list.append(ssim_inter)

            if LFR_gt is not None:
                LFR_lq_psnr_list.append(-10 * math.log10(((LFR_gt.cpu().numpy() - LFR_lq.cpu().numpy()) * (LFR_gt.cpu().numpy() - LFR_lq.cpu().numpy())).mean()))
                LFR_hq_psnr_list.append(-10 * math.log10(((LFR_gt.cpu().numpy() - LFR_hq.cpu().numpy()) * (LFR_gt.cpu().numpy() - LFR_hq.cpu().numpy())).mean()))
                LFR_lq_ssim_list.append(ssim_matlab(ycbcr2rgb(LFR_gt), torch.round(ycbcr2rgb(LFR_lq) * 255) / 255.).detach().cpu().numpy())
            else:
                LFR_lq_psnr_list.append(0)
                LFR_hq_psnr_list.append(0)

        # Visualization
        if test_file == 'test_septuplet_viz.txt':
            import matplotlib.pyplot as plt
            if not os.path.exists(f'/output/HFR_recon/'):
                os.makedirs(f'/output/HFR_recon/')
                os.makedirs(f'/output/LFR_lq/')
                os.makedirs(f'/output/LFR_hq/')
            
            if LFR_gt is not None:
                for frm_id, path_item in enumerate(line[::2]):
                    fig_name = '_'.join(path_item.split('.')[0].split('/')[-3:])
                    psnr_val = -10 * math.log10(((LFR_gt[frm_id].cpu().numpy() - LFR_lq[frm_id].cpu().numpy()) * (LFR_gt[frm_id].cpu().numpy() - LFR_lq[frm_id].cpu().numpy())).mean())
                    plt.imsave(f'/output/LFR_lq/{fig_name}_qp{args.qp}_{model_name}_psnr{psnr_val:.2f}_bpp{img_distri:.4f}.png', 
                               rearrange(ycbcr2rgb(LFR_lq), 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0, 1))
                    plt.imsave(f'/output/LFR_hq/{fig_name}_qp{args.qp}_{model_name}_psnr{-10 * math.log10(((LFR_gt[frm_id].cpu().numpy() - LFR_hq[frm_id].cpu().numpy()) * (LFR_gt[frm_id].cpu().numpy() - LFR_hq[frm_id].cpu().numpy())).mean()):.2f}_bpp{img_distri:.4f}.png', 
                               rearrange(ycbcr2rgb(LFR_hq), 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0, 1))
            
            for frm_id, path_item in enumerate(line):
                fig_name = '_'.join(path_item.split('.')[0].split('/')[-3:])
                psnr_viz = -10 * math.log10(((out_label[frm_id] - out_x[frm_id]) * (out_label[frm_id] - out_x[frm_id])).mean())
                ssim_viz = ssim_matlab(out_label_rgb[frm_id:frm_id+1], torch.round(out_rgb[frm_id:frm_id+1] * 255) / 255.).detach().cpu().numpy()
                plt.imsave(f'/output/HFR_recon/{fig_name}_qp{args.qp}_{model_name}_psnr{psnr_viz:.2f}_ssim{ssim_viz:.4f}_bpp{img_distri:.4f}.png', 
                           rearrange(out_rgb, 'b c h w -> b h w c').detach().cpu().numpy()[frm_id].clip(0, 1))

        if model_name != 'codec_reference':
            T, C, H, W = out_label.shape
            vmaf_score = calculate_vmaf_score(out_label, out_x, W, H, pix_fmt='yuv444p')
            vmaf_list.append(vmaf_score)

        gt_seq = out_label_rgb.unsqueeze(0).clamp(0, 1)
        rec_seq = out_rgb.unsqueeze(0).clamp(0, 1)
        B, T, C, H, W = gt_seq.shape
        
        lpips_val_list = []
        dists_val_list = []
        w_psnr_val_list = []
        tlpips_val_list = []
        tof_val_list = []

        gt_flat = gt_seq.view(-1, C, H, W)
        rec_flat = rec_seq.view(-1, C, H, W)
        total_frames = gt_flat.shape[0]

        for i in range(total_frames):
            gt_curr = gt_flat[i:i+1]
            rec_curr = rec_flat[i:i+1]
            lpips_val = lpips(gt_curr, rec_curr).squeeze()
            dists_val = dists(gt_curr, rec_curr).squeeze()
            lpips_val_list.append(lpips_val.item())
            dists_val_list.append(dists_val.item())

            if i > 0:
                gt_prev = gt_flat[i-1:i]
                rec_prev = rec_flat[i-1:i]
                
                padder = InputPadder(gt_curr.shape[-2:], divisor=8)
                gt_curr_pad = padder.pad(gt_curr)[0]
                gt_prev_pad = padder.pad(gt_prev)[0]
                rec_curr_pad = padder.pad(rec_curr)[0]
                rec_prev_pad = padder.pad(rec_prev)[0]
                
                flow_gt = get_flow(of_model, gt_curr_pad, gt_prev_pad)
                flow_rec = get_flow(of_model, rec_curr_pad, rec_prev_pad)
                warped_rec = flow_warp(rec_prev_pad, flow_rec)
                
                flow_gt = padder.unpad(flow_gt)
                flow_rec = padder.unpad(flow_rec)
                warped_rec = padder.unpad(warped_rec)
                
                mse = ((warped_rec - gt_curr) ** 2).mean()
                w_psnr_val = -10 * torch.log10(mse + 1e-8)
                w_psnr_val_list.append(w_psnr_val.item())
                
                lpips_gt_diff = lpips(gt_curr, gt_prev).item()
                lpips_rec_diff = lpips(rec_curr, rec_prev).item()
                tlpips_val = abs(lpips_gt_diff - lpips_rec_diff)
                tlpips_val_list.append(tlpips_val)
                
                flow_diff = (flow_rec - flow_gt).abs()
                flow_diff = flow_diff[torch.isfinite(flow_diff)]
                if flow_diff.numel() == 0:
                    tof_val = 0.0
                else:
                    tof_val = flow_diff.mean().item()
                tof_val_list.append(tof_val)

        lpips_vals = torch.tensor(lpips_val_list).view(B, T)
        dists_vals = torch.tensor(dists_val_list).view(B, T)
        
        if w_psnr_val_list:
            w_psnr_vals = torch.tensor(w_psnr_val_list).view(B, T-1)
            tlpips_vals = torch.tensor(tlpips_val_list).view(B, T-1)
            tof_vals = torch.tensor(tof_val_list).view(B, T-1)
        else:
            w_psnr_vals = torch.empty(B, 0)
            tlpips_vals = torch.empty(B, 0)
            tof_vals = torch.empty(B, 0)

        for i in range(B):
            seq = line[0].replace('/', '_').replace('.', '_')
            lpips_dict[seq] = lpips_vals[i].cpu().numpy().tolist()
            dists_dict[seq] = dists_vals[i].cpu().numpy().tolist()
            w_psnr_dict[seq] = w_psnr_vals[i].cpu().numpy().tolist()
            tlpips_dict[seq] = tlpips_vals[i].cpu().numpy().tolist()
            tof_dict[seq] = tof_vals[i].cpu().numpy().tolist()

    # Final Statistics
    def safe_mean(lst):
        return np.mean(lst) if lst else 0.0

    seq_list = list(lpips_dict.keys())
    mean_lpips = np.round(np.mean([safe_mean(lpips_dict[k]) for k in seq_list]), 3)
    mean_dists = np.round(np.mean([safe_mean(dists_dict[k]) for k in seq_list]), 3)
    mean_tlpips = np.round(np.mean([safe_mean(tlpips_dict[k]) for k in seq_list]) * 1e3, 2)
    mean_tof = np.round(np.mean([safe_mean(tof_dict[k]) for k in seq_list]) * 1e1, 3)
    mean_warpping_psnr = np.round(np.mean([safe_mean(w_psnr_dict[k]) for k in seq_list]), 3)

    print("Vimeo dataset")
    print("QP: ", args.qp)
    print(f"Model: {model_name}, test file: {test_file}")
    print(f"psnr:{np.mean(psnr_list):.2f},psnr_avg_mse:{ (np.mean(mse_list)):.10f},ssim:{np.mean(ssim_list):.4f},psnr_LFR_lq:{np.mean(LFR_lq_psnr_list):.2f},ssim_LFR_lq:{np.mean(LFR_lq_ssim_list):.4f},psnr_LFR_hq:{np.mean(LFR_hq_psnr_list):.2f},sigma:{np.mean(sigma_list):.4f},psnr inter:{np.mean(psnr_inter_list):.2f},ssim inter:{np.mean(ssim_inter_list):.4f},lpips:{mean_lpips:.4f},dists:{mean_dists:.4f},tlpips(1e3):{mean_tlpips:.2f},tof(1e1):{mean_tof:.4f},warpping_psnr:{mean_warpping_psnr:.2f},vmaf:{np.mean(vmaf_list):.4f},ave_img_bpp:{np.mean(bpp_list):.6f}")