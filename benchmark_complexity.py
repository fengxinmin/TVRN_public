import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
import time
import gc
import json
import shutil
import tempfile
import subprocess
import threading

try:
    from tqdm import tqdm
    tqdm_open = True
except ImportError:
    tqdm_open = False

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
import yaml
import psutil
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from DISTS_pytorch import DISTS
from torchvision.transforms import ToTensor

sys.path.append('.')
from benchmark.utils.pytorch_msssim import ssim_matlab
from utils.functions import ycbcr2rgb, rgb2ycbcr
from RAFT.core.raft import RAFT as raft
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
from benchmark.utils.padder import InputPadder
import options.options as option
from models.VRN_model import TVRNCodecModel as Model
from EBME.bi_flownet import BiFlowNet
from EBME.fusionnet import FusionNet
from models.VRN_model import STAAModel
from models.modules.STDR_Net import Net as STDR_Net
from XVFI.XVFInet import XVFInet
from torch.nn.parallel import DataParallel, DistributedDataParallel

vmaf_root = '/tmp/vmaf-1.3.15'
os.environ['PYTHONPATH'] = f"{vmaf_root}/libsvm/python:{vmaf_root}/python/src:{vmaf_root}:" + os.environ.get('PYTHONPATH', '')

process = psutil.Process(os.getpid())
BASE_OUTPUT_DIR = "/data/fengxm/vimeo90k/tvrn_revision"


def calculate_vmaf_score(ref_np, dist_np, width, height, pix_fmt='yuv444p'):
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
    except Exception:
        return None
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(dist_path):
            os.remove(dist_path)


def clear_gpu_memory(device_id=0):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)


def get_flow(of_model, target, source, rescale_factor=1):
    flows = of_model(target, source)
    flow = flows[-1]
    if rescale_factor != 1:
        flow = F.interpolate(flow // rescale_factor, scale_factor=1 / rescale_factor, mode='bilinear')
    return flow.permute(0, 2, 3, 1)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float().type_as(x)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=False)


def compute_flow_magnitude(flow):
    return flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2


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
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--name', default='test_vfiformer', type=str)
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--net_name', default='VFIformer', type=str)
parser.add_argument('--window_size', default=8, type=int)
parser.add_argument('--module_scale_factor', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--data_root', default='/home/liyinglu/newData/datasets/vfi/SNU-FILM/', type=str)
parser.add_argument('--testset', default='FILM', type=str)
parser.add_argument('--test_level', default='extreme', type=str)
parser.add_argument('--crop_size', default=192, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_augmentation', default=False, type=bool)
parser.add_argument('--resume', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
parser.add_argument('--resume_flownet', default='', type=str)
parser.add_argument('--save_folder', default='./test_results', type=str)
parser.add_argument('--save_result', action='store_true')
parser.add_argument('-opt', type=str)
parser.add_argument('-path', type=str, required=True)
parser.add_argument('-mode', type=str, default='crf')
parser.add_argument('-model', type=str, default='TVRN')
parser.add_argument('-qp', type=int, default=27)
parser.add_argument('--datasets', type=str, default='perceptual_video')
parser.add_argument('--expdir', type=str, default='/data/fengxm/vimeo90k/pretrained_model/VFIPS/exp/eccv_ms_multiscale_v33/')
parser.add_argument('--depth_ksize', type=int, default=1)
parser.add_argument('--flow', type=str2bool, default=False)
parser.add_argument('--autodata', type=str2bool, default=True)
parser.add_argument('--norm', type=str, default='sigmoid')
parser.add_argument('--checkpoints', type=str, default=None)
parser.add_argument('-staa_opt', type=str)
parser.add_argument('--codec_type', type=str, default='hevc', choices=['avc', 'hevc', 'av1', 'vp9', 'vvc'])
parser.add_argument('--slice_id', type=int, default=0)
parser.add_argument('--total_slices', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--small', default=False)
parser.add_argument('--mixed_precision', default=False)
parser.add_argument('--alternate_corr', default=False)
parser.add_argument('--record_time', default=False)
parser.add_argument('--cpu', default=False)

args = parser.parse_args()

if args.model == 'TVRN_S':
    args.opt = '/code/codes/options/test_septuplet/test_TVRN_without_restoration.yml'
    args.checkpoints = '/model/fengxm/VRN/MIMO_VRN/checkpoints/model_wo_restoration.pth'

opt = option.parse(args.opt, is_train=False)
opt['codec_type'] = args.codec_type

if args.checkpoints is not None:
    opt['path']['pretrain_model_G'] = args.checkpoints

torch.cuda.set_device(args.gpu_id)
device = torch.device('cpu') if args.cpu else torch.device(f"cuda:{args.gpu_id}")
opt['gpu_ids'] = None if args.cpu else args.gpu_id

model_name = args.model
LFR_gt, LFR_hq, LFR_lq = None, None, None
TTA = True
average_metric = True

if 'TVRN' in model_name:
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
    load_net = torch.load("/model/fengxm/VRN/UPR_Net/pretrained/upr-base.pkl")
    load_net_clean = {k[7:] if k.startswith('module.') else k: v for k, v in load_net.items()}
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.to(device)
elif model_name == 'UPR_l':
    from models.VFI_model import UPRModelLarge
    model = UPRModelLarge()
    load_net = torch.load("/model/fengxm/VRN/UPR_Net/pretrained/upr-large.pkl")
    load_net_clean = {k[7:] if k.startswith('module.') else k: v for k, v in load_net.items()}
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.to(device)
elif model_name == 'UPR_L':
    from models.VFI_model import UPRModelLLarge
    model = UPRModelLLarge()
    load_net = torch.load("/model/fengxm/VRN/UPR_Net/pretrained/upr-llarge.pkl")
    load_net_clean = {k[7:] if k.startswith('module.') else k: v for k, v in load_net.items()}
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.to(device)
elif model_name == 'IFRNet':
    from models.IFRNet_L import Model
    model = Model()
    load_net = torch.load("/data/fengxm/vimeo90k/pretrained_model/IFRNet/IFRNet_L_Vimeo90K.pth")
    load_net_clean = {k[7:] if k.startswith('module.') else k: v for k, v in load_net.items()}
    model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    model.to(device)
elif model_name == 'EBME':
    def load_pretrained_state_dict(module, module_name, module_args):
        load_pretrain = getattr(module_args, "load_pretrain", True)
        if not load_pretrain:
            return False
        model_file = getattr(module_args, "model_file", "")
        if not model_file or not os.path.exists(model_file):
            raise ValueError(f"Pretrained file missing for {module_name}")
        pretrained_state_dict = torch.load(model_file)
        return {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}

    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = 5
    bi_flownet_args.load_pretrain = True
    bi_flownet_args.model_file = '/data/fengxm/vimeo90k/pretrained_model/EBME/ebme/bi-flownet.pkl'

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = False
    fusionnet_args.load_pretrain = True
    fusionnet_args.model_file = '/data/fengxm/vimeo90k/pretrained_model/EBME/ebme/fusionnet.pkl'

    bi_flownet = BiFlowNet(bi_flownet_args).to(device)
    fusionnet = FusionNet(fusionnet_args).to(device)
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
    model = model.to(device).eval()
elif model_name in ['GIMM', 'GIMM+VQE']:
    from models_GIMM.models import create_model
    from argparse import Namespace

    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return Namespace(**d)

    with open('/code/codes/models_GIMM/gimmvfi_r_arb.yaml', 'r') as f:
        gimm_config = dict_to_namespace(yaml.safe_load(f))
    gimm_config.arch.fwarp_type = 'linear'
    gimm_config.arch.normalize_weight = True
    model, _ = create_model(gimm_config.arch)
    model = model.to(device)
    ckpt = torch.load('/data/fengxm/vimeo90k/pretrained_model/GIMM/gimmvfi_r_arb.pt', map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    if model_name == 'GIMM+VQE':
        restoration_model = STDR_Net(opt['network_STDR'], type='STDR_both_adaptor')
        load_path_G = opt['path']['pretrain_model_G']
        if load_path_G:
            load_net = torch.load(load_path_G)
            load_net_clean = {}
            for k, v in load_net.items():
                if k.startswith('module.restoration_module.'):
                    load_net_clean[k[len('module.restoration_module.'):]] = v
                elif k.startswith('restoration_module.'):
                    load_net_clean[k[len('restoration_module.'):]] = v
            restoration_model.load_state_dict(load_net_clean, strict=True)

        from models.modules.Inv_arch import Ranker_wo_res
        ranker = Ranker_wo_res(in_chans=3, out_chans=1)
        load_path_ranker = opt['path']['pretrain_ranker']
        if load_path_ranker:
            load_net = torch.load(load_path_ranker)
            load_net_clean = {}
            for k, v in load_net.items():
                if k.startswith('module.ranker.'):
                    load_net_clean[k[len('module.'):]] = v
                else:
                    load_net_clean[k] = v
            ranker.load_state_dict(load_net_clean, strict=True)

elif model_name == 'CVRS':
    torch.backends.cudnn.enabled = False
    from models_cvrs.arch.IMSM import IND_inv3D
    from models_cvrs.utils.options import yaml_load

    inv_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/config.yml')['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight = torch.load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/model.pth')
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
else:
    raise Exception('invalid model name')


def monitor_memory(process, interval, result):
    max_mem = 0
    while not result['stop']:
        mem = process.memory_info().rss
        if mem > max_mem:
            max_mem = mem
        time.sleep(interval)
    result['max_memory'] = max_mem


if __name__ == '__main__':
    monitor_interval = 0.1
    monitor_result = {'max_memory': 0, 'stop': False}
    monitor_thread = threading.Thread(target=monitor_memory, args=(process, monitor_interval, monitor_result))
    monitor_thread.start()

    Quantization_H265_Stream = Quantization_H265_Stream(args.qp, -1, None, opt)

    downscaling_latency_list = []
    upscaling_latency_list = []
    encoding_latency_list = []
    decoding_latency_list = []

    try:
        out_label = group_1 = torch.randn(7, 3, 720, 1080).to(device)
        img1 = img2 = img3 = img4 = img5 = img6 = img7 = torch.randn(720, 1080, 3).to(device)

        if 'TVRN' in model_name:
            gt = out_label[:3]
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)
            end.record()
            torch.cuda.synchronize()
            downscaling_latency = start.elapsed_time(end)
            peak_mem_encoder = torch.cuda.max_memory_allocated() / 1024**3

            start = time.time()
            Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()
            end = time.time()
            encoding_latency = (end - start) * 1000

            start = time.time()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:, [0, 2, 1]]
            img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]
            end = time.time()
            decoding_latency = (end - start) * 1000

            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).to(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name in ['IFRNet']:
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            downscaling_latency = 0

            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')

            start_time = time.time()
            Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()
            encoding_latency = (time.time() - start_time) * 1000
            peak_mem_encoder = 0

            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].to(device), v_seg[1:2].to(device), v_seg[2:3].to(device), v_seg[3:4].to(device)
            end_time = time.time()
            decoding_latency = (end_time - start_time) * 1000

            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            padder = InputPadder(img1_lq.shape, divisor=20)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]

            pred_list = []
            for i in range(3):
                embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(device)
                pred = model.inference(group_list[i], group_list[i + 1], embt, scale_factor=0.8)
                pred_list.append(pred)

            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name == 'EBME':
            downscaling_latency = 0
            peak_mem_encoder = 0
            start_time = time.time()

            Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))

            encoding_latency = (time.time() - start_time) * 1000
            start_time = time.time()

            _, img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].to(device), v_seg[1:2].to(device), v_seg[2:3].to(device), v_seg[3:4].to(device)
            padder = InputPadder(img1_lq.shape, divisor=64)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []

            decoding_latency = (time.time() - start_time) * 1000
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for i in range(3):
                bi_flow = bi_flownet(group_list[i], group_list[i + 1])
                pred = fusionnet(group_list[i], group_list[i + 1], bi_flow, time_period=0.5)
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name == 'RIFE':
            downscaling_latency = 0
            peak_mem_encoder = 0
            start_time = time.time()

            Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()

            encoding_latency = (time.time() - start_time) * 1000
            start_time = time.time()

            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].to(device), v_seg[1:2].to(device), v_seg[2:3].to(device), v_seg[3:4].to(device)
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []

            decoding_latency = (time.time() - start_time) * 1000
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for i in range(3):
                pred = model.inference(group_list[i], group_list[i + 1])
                pred = padder.unpad(pred)
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(pred)
                pred_list.append(pred)
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name in ['GIMM', 'GIMM+VQE']:
            downscaling_latency = 0
            peak_mem_encoder = 0
            start_time = time.time()

            Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
            Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))

            encoding_latency = (time.time() - start_time) * 1000
            start_time = time.time()

            _, img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].to(device), v_seg[1:2].to(device), v_seg[2:3].to(device), v_seg[3:4].to(device)
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []

            decoding_latency = (time.time() - start_time) * 1000
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for i in range(3):
                I0, I2 = group_list[i], group_list[i + 1]
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
                pred = all_outputs["imgt_pred"][0]
                out_list.append(padder.unpad(group_list[i]))
                out_list.append(padder.unpad(pred))
                pred_list.append(padder.unpad(pred))
            out_list.append(padder.unpad(group_list[-1]))
            pred = torch.cat(pred_list, dim=0)
            out_x = torch.cat(out_list, dim=0)

            LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
            LFR_lq = v_seg.to(device)
            LFR_hq = out_x[[0, 2, 4, 6]]

            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name == 'STAA':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]

            LF_processed = []
            for i in range(0, group_1_pad.shape[0] - 1, 2):
                LF = model.netG(x=group_1_pad[i:i + 3].unsqueeze(0), rev=False)[0]
                LF_processed.append(LF[0])
                group_1_pad[i + 2] = LF[-1]
            LF_processed.append(LF[-1])
            LF_processed = torch.stack(LF_processed, dim=0)
            LF = padder.unpad(LF_processed)

            end.record()
            torch.cuda.synchronize()
            downscaling_latency = start.elapsed_time(end)
            start_time = time.time()
            peak_mem_encoder = torch.cuda.max_memory_allocated() / 1024**3

            Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]].unsqueeze(0), 'b t c h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()

            encoding_latency = (time.time() - start_time) * 1000
            start_time = time.time()

            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:, [0, 2, 1]]
            img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1], v_seg[1:2], v_seg[2:3], v_seg[3:4]
            padder = InputPadder(img1_lq.shape, divisor=32)
            img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
            group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
            pred_list = []
            out_list = []

            decoding_latency = (time.time() - start_time) * 1000
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for i in range(3):
                recon_3_frames = model.netG(x=[torch.stack([group_list[2 - i].to(device), group_list[3 - i].to(device)], dim=1), None], rev=True)
                out_list.append(padder.unpad(recon_3_frames[:, -1]))
                out_list.append(padder.unpad(recon_3_frames[:, -2]))
                pred_list.append(padder.unpad(recon_3_frames[:, -2]))
                group_list[2 - i] = recon_3_frames[:, -3]
            out_list.append(padder.unpad(recon_3_frames[:, -3]))
            out_x = torch.cat(out_list[::-1], dim=0)
            pred = torch.cat(pred_list[::-1], dim=0)

            LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.cat([img1_lq, img2_lq, img3_lq, img4_lq], dim=0).to(device)
            LFR_hq = out_x[[0, 2, 4, 6]]

            end.record()
            torch.cuda.synchronize()
            upscaling_latency = start.elapsed_time(end)
            peak_mem_decoder = torch.cuda.max_memory_allocated() / 1024**3

        elif model_name == 'CVRS':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
            gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
            padder = InputPadder(group_1.shape, divisor=64)
            group_1_pad = padder.pad(group_1)[0]

            down_size = (4, group_1_pad.shape[-2], group_1_pad.shape[-1])
            x_down = rescale_model.inference_down(rearrange(ycbcr2rgb(group_1_pad), 't c h w -> c t h w').unsqueeze(0), down_size)
            LR_img = model.inference_latent2RGB(x_down)
            LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> b t c h w')[0]).unsqueeze(0), 'b t c h w -> b c t h w')

            end.record()
            torch.cuda.synchronize()
            downscaling_latency = start.elapsed_time(end)
            start_time = time.time()

            Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
            Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
            _, img_distri = Quantization_H265_Stream.close_writer()
            Quantization_H265_Stream.open_reader(verbosity=0)
            v_seg = Quantization_H265_Stream.read_multi_frames(4)
            v_seg = v_seg[:, [0, 2, 1]]
            img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]

            rev_back = model.inference_RGB2latent(rearrange(ycbcr2rgb(v_seg).unsqueeze(0).to(device), 'b t c h w -> b c t h w'))
            out_x = rescale_model.inference_up(rev_back, (group_1_pad.shape[0], group_1_pad.shape[-2], group_1_pad.shape[-1]))
            out_x = padder.unpad(out_x)
            out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
            out_x = rgb2ycbcr(out_x)
            pred = out_x[1::2]

            LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
            LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).to(device)
            LFR_lq = padder.unpad(LFR_lq)
            LFR_hq = out_x[[0, 2, 4, 6]]
        else:
            raise Exception('invalid model name')

        downscaling_latency_list.append(downscaling_latency)
        upscaling_latency_list.append(upscaling_latency)
        encoding_latency_list.append(encoding_latency)
        decoding_latency_list.append(decoding_latency)

    finally:
        monitor_result['stop'] = True
        monitor_thread.join(timeout=2)
        peak_memory_bytes = monitor_result['max_memory']

    print(f"Model:{model_name},QP:{args.qp}", end=" ")
    print(f"downscaling_latency: {np.mean(downscaling_latency_list)/1000:.2f} s/clip", end=" ")
    print(f"encoding_latency: {np.mean(encoding_latency_list)/1000:.2f} s/clip", end=" ")
    print(f"decoding_latency: {np.mean(decoding_latency_list)/1000:.2f} s/clip", end=" ")
    print(f"upscaling_latency: {np.mean(upscaling_latency_list)/1000:.2f} s/clip", end=" ")
    total_latency = np.sum([np.mean(encoding_latency_list), np.mean(decoding_latency_list), np.mean(downscaling_latency_list), np.mean(upscaling_latency_list)]) / 1000
    print(f"end-to-end latency: {total_latency:.2f} s/clip", end=" ")
    print(f"peak_memory_gpu(sender/receiver): {peak_mem_encoder:.2f}/{peak_mem_decoder:.2f} GB", end=" ")
    print(f"peak_memory_cpu: {peak_memory_bytes / 1024**3:.2f} GB")