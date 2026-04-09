import os
import sys
import cv2
import math
import argparse
import warnings
import numpy as np
import multiprocessing
from tqdm import tqdm

warnings.filterwarnings('ignore')

from einops import rearrange
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
from benchmark.utils.padder import InputPadder
from collections import OrderedDict
import options.options as option
from models.VRN_model import TVRNCodecModel as Model
from utils.functions import ycbcr444_to_420
from PIL import Image
from EBME.bi_flownet import BiFlowNet
from EBME.fusionnet import FusionNet
from XVFI.XVFInet import XVFInet
from models.VRN_model import STAAModel
from models.modules.STDR_Net import Net as STDR_Net
import yaml
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
torch.set_grad_enabled(False)
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

sys.path.append('.')
from benchmark.utils.pytorch_msssim import ssim_matlab
from utils.functions import ycbcr2rgb, rgb2ycbcr

import tempfile
import subprocess
import json
vmaf_root = '/tmp/vmaf-1.3.15'
os.environ['PYTHONPATH'] = f"{vmaf_root}/libsvm/python:{vmaf_root}/python/src:{vmaf_root}:" + os.environ.get('PYTHONPATH', '')

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
    except Exception as e:
        print(f"VMAF calculation failed: {e}")
        print(f"Stderr: {result.stderr if 'result' in locals() else 'N/A'}")
        return None
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(dist_path):
            os.remove(dist_path)


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


class VideoSequenceDataset(Dataset):
    def __init__(self, base_path, file_list, device):
        self.base_path = base_path
        self.file_list = file_list
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        line = self.file_list[idx]
        seq_path = os.path.join(self.base_path, 'sequences', line)
        processed_frames = []
        for i in range(1, 8):
            img_path = os.path.join(seq_path, f'im{i}.png')
            img_bgr = cv2.imread(img_path)
            img_rgb = img_bgr[:, :, [2, 1, 0]].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(rearrange(img_rgb, 'h w c -> c h w')).float()
            yuv_tensor = rgb2ycbcr(img_tensor.unsqueeze(0)).squeeze(0)
            img_yuv_hwc = rearrange(yuv_tensor, 'c h w -> h w c')
            processed_frames.append(img_yuv_hwc)
        return {
            'line_name': line,
            'img1': processed_frames[0],
            'img2': processed_frames[1],
            'img3': processed_frames[2],
            'img4': processed_frames[3],
            'img5': processed_frames[4],
            'img6': processed_frames[5],
            'img7': processed_frames[6],
            'seq_path': seq_path
        }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from DISTS_pytorch import DISTS
from torchvision.transforms import ToTensor
from RAFT.core.raft import RAFT as raft
import gc


def clear_gpu_memory(device_id=0):
    gc.collect()
    torch.cuda.empty_cache()
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


if __name__ == '__main__':
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
    args = parser.parse_args()

    if args.model == 'TVRN_S':
        args.opt = '/code/codes/options/test_septuplet/test_TVRN_without_restoration.yml'
        args.checkpoints = '/model/fengxm/VRN/MIMO_VRN/checkpoints/model_wo_restoration.pth'

    opt = option.parse(args.opt, is_train=False)
    opt['codec_type'] = args.codec_type

    if args.checkpoints is not None:
        opt['path']['pretrain_model_G'] = args.checkpoints

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    BASE_OUTPUT_DIR = "/data/fengxm/vimeo90k/tvrn_revision"

    model_name = args.model
    TTA = True
    average_metric = True

    if 'TVRN' in model_name:
        model = Model(opt)
    elif model_name == 'STAA':
        staa_opt = option.parse(args.staa_opt, is_train=False)
        model = STAAModel(staa_opt, device)
    elif model_name == 'EMA':
        import config as cfg
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
        from Trainer import Model
        model = Model(-1)
        model.load_model(path=opt['path']['EMA_model'])
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
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=True)
        model.eval()
        model.to(device)
    elif model_name == 'UPR_l':
        from models.VFI_model import UPRModelLarge
        model = UPRModelLarge()
        load_net = torch.load("/model/fengxm/VRN/UPR_Net/pretrained/upr-large.pkl")
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=True)
        model.eval()
        model.to(device)
    elif model_name == 'UPR_L':
        from models.VFI_model import UPRModelLLarge
        model = UPRModelLLarge()
        load_net = torch.load("/model/fengxm/VRN/UPR_Net/pretrained/upr-llarge.pkl")
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=True)
        model.eval()
        model.to(device)
    elif model_name == 'IFRNet':
        from models.IFRNet_L import Model
        model = Model()
        load_net = torch.load("/data/fengxm/vimeo90k/pretrained_model/IFRNet/IFRNet_L_Vimeo90K.pth")
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=True)
        model.eval()
        model.to(device)
    elif model_name == 'RIFE':
        from models_RIFE.train_log.RIFE_HDv3 import Model
        model = Model(device=device)
        model.load_model('/code/codes/models_RIFE/train_log', -1)
        model.eval()
        model.device()
    elif model_name == 'VFIformer':
        from models_VFIformer.modules import define_G
        args.test_level = 'medium'
        args.net_name = 'VFIformer'
        args.resume = '/data/fengxm/vimeo90k/pretrained_model/VFIformer/net_220.pth'
        args.dist = False
        args.gpu_ids = [0, ]
        model = define_G(args)
        load_path = args.resume
        network = model
        load_net = torch.load(load_path, map_location=torch.device('cpu'))
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=True)
        down_scale = 2
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
    elif model_name == 'VideoINR':
        import models_videoINR.models.modules.Sakuya_arch as Sakuya_arch
        model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
        model.load_state_dict(torch.load('/data/fengxm/vimeo90k/pretrained_model/videoINR/latest_G.pth'), strict=True)
        model.eval()
        model = model.to(device)

        def single_forward(model, imgs_in, space_scale, time_scale):
            with torch.no_grad():
                b, n, c, h, w = imgs_in.size()
                h_n = int(4 * np.ceil(h / 4))
                w_n = int(4 * np.ceil(w / 4))
                imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
                imgs_temp[:, :, :, 0:h, 0:w] = imgs_in
                time_Tensors = [torch.tensor([i / time_scale])[None].to(device) for i in range(time_scale)]
                model_output = model(imgs_temp, time_Tensors, space_scale, test=True)
                return model_output
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
        model.to(device)
        model.eval()
    elif model_name == 'CVRS':
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
    elif model_name == 'CVRS_finetuned':
        from models.VRN_model import CSTVRModel as Model
        opt['path']['pretrain_model_G'] = '/model/fengxm/VRN/MIMO_VRN/CSTVR_w_surrogate_net/70000_G.pth'
        model = Model(opt)
    else:
        raise Exception('invalid model name')

    LFR_gt, LFR_hq, LFR_lq = None, None, None

    print(f'=========================Starting testing=========================')
    print(f'Dataset: Vimeodataset Model: {model_name} TTA: {TTA}')

    path = args.path
    dirs = os.listdir(path)
    level_list = ['sep_testlist.txt',]

    def custom_collate_fn(batch):
        elem = batch[0]
        ret = {}
        for key in elem:
            if key == 'line_name' or key == 'seq_path':
                ret[key] = [d[key] for d in batch]
            else:
                ret[key] = torch.stack([d[key] for d in batch], dim=0)
        return ret

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
        psnr_LFR = []
        LFR_lq_psnr_list, LFR_lq_ssim_list, LFR_hq_psnr_list = [], [], []
        vmaf_list = []

        with open(os.path.join(path, test_file), "r") as f:
            for line in f:
                line = line.strip()
                file_list.append(line)

        if args.total_slices > 1:
            total_len = len(file_list)
            slice_size = (total_len + args.total_slices - 1) // args.total_slices
            start_idx = args.slice_id * slice_size
            end_idx = min(start_idx + slice_size, total_len)
            file_list = file_list[start_idx:end_idx]
            print(f"[GPU {args.gpu_id}] Processing slice {args.slice_id}/{args.total_slices}: items {start_idx} to {end_idx-1}")
        else:
            print(f"[GPU {args.gpu_id}] Processing full list.")

        output_folder = os.path.join(BASE_OUTPUT_DIR, args.codec_type, 'vimeo90k', args.model, f"QP{args.qp}")
        output_folder_in = args.path
        output_folder_out = os.path.join(output_folder, 'output')

        dataset = VideoSequenceDataset(path, file_list, device=device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

        lpips = LPIPS(reduction='mean').to(device)
        dists = DISTS().to(device)

        iterator = dataloader

        with torch.no_grad():
            for batch_data in iterator:
                line = batch_data['line_name'][0]
                seq_path = batch_data['seq_path'][0]

                img1 = batch_data['img1'][0].cuda(device)
                img2 = batch_data['img2'][0].cuda(device)
                img3 = batch_data['img3'][0].cuda(device)
                img4 = batch_data['img4'][0].cuda(device)
                img5 = batch_data['img5'][0].cuda(device)
                img6 = batch_data['img6'][0].cuda(device)
                img7 = batch_data['img7'][0].cuda(device)

                if 'TVRN' in model_name:
                    group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    padder = InputPadder(group_1.shape, divisor=64)
                    group_1_pad = padder.pad(group_1)[0]
                    LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)

                    Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    v_seg = v_seg[:, [0, 2, 1]]
                    img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]

                    out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
                    out_x = padder.unpad(out_x)
                    out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
                    pred = out_x[1::2]

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
                    LFR_lq = padder.unpad(LFR_lq)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'STAA':
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

                    Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]].unsqueeze(0), 'b t c h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    v_seg = v_seg[:, [0, 2, 1]]
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1], v_seg[1:2], v_seg[2:3], v_seg[3:4]
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        recon_3_frames = model.netG(x=[torch.stack([group_list[2 - i].cuda(device), group_list[3 - i].cuda(device)], dim=1), None], rev=True)
                        out_list.append(padder.unpad(recon_3_frames[:, -1]))
                        out_list.append(padder.unpad(recon_3_frames[:, -2]))
                        pred_list.append(padder.unpad(recon_3_frames[:, -2]))
                        group_list[2 - i] = recon_3_frames[:, -3]
                    out_list.append(padder.unpad(recon_3_frames[:, -3]))
                    out_x = torch.cat(out_list[::-1], dim=0)
                    pred = torch.cat(pred_list[::-1], dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = torch.cat([img1_lq, img2_lq, img3_lq, img4_lq], dim=0).cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'EMA':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred = model.inference(group_list[i], group_list[i + 1], TTA=TTA, fast_TTA=TTA)[0]
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred.unsqueeze(0))
                        pred_list.append(pred.unsqueeze(0))
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = v_seg.cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'SGM':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred = model.hr_inference(group_list[i], group_list[i + 1], TTA=TTA, down_scale=1, fast_TTA=False).clamp(0.0, 1.0)
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                elif model_name in ['UPR', 'UPR_l', 'UPR_L']:
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred, _, _ = model(group_list[i], group_list[i + 1], time_period=0.5)
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                elif model_name in ['IFRNet']:
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).cuda(device)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = (img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        img1_lq_pad2, img2_lq_pad2 = (group_list[i], group_list[i + 1])
                        pred = model.inference(img1_lq_pad2, img2_lq_pad2, embt=embt)
                        out_list.append((group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append((group_list[-1]))
                    out_x = torch.cat(out_list, dim=0)
                    pred = torch.cat(pred_list, dim=0)

                elif model_name in ['codec_reference']:
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
                    LF = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(7)
                    out_x = torch.tensor(v_seg).cuda(device)
                    pred = out_x[1::2]

                elif model_name == 'VFIformer':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=64)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        down_scale = 0.5
                        img1_down = F.interpolate(group_list[i], scale_factor=down_scale, mode="bilinear", align_corners=False)
                        img3_down = F.interpolate(group_list[i + 1], scale_factor=down_scale, mode="bilinear", align_corners=False)
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
                        flow_down = model.get_flow(img1_down.cuda(device), img3_down.cuda(device))
                        if h % 64 != 0 or w % 64 != 0:
                            flow_down = flow_down[:, :, :h, :w]
                        flow = F.interpolate(flow_down, scale_factor=1 / down_scale, mode="bilinear", align_corners=False) * 1 / down_scale
                        pred, _, = model(img1_lq_pad.cuda(device), img2_lq_pad.cuda(device), flow_pre=flow)
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                elif model_name == 'EBME':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=64)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

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

                elif model_name == 'XVFI':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred = model(torch.stack([group_list[i], group_list[i + 1]], dim=2), t_value=torch.tensor(0.5).reshape(group_list[i].shape[0], 1).to(group_list[i].device), is_training=False)
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                elif model_name in ['GIMM', 'GIMM+VQE']:
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)

                    if model_name == 'GIMM+VQE':
                        x = rearrange(v_seg.unsqueeze(0), 'b t c h w -> b c t h w')
                        nfs = x.shape[2]
                        output_data = []
                        for idx in range(nfs):
                            if opt['network_G']['compensation_module'] == 'STDR_both_adaptor':
                                f_ranker = ranker(x[:, :, idx], out_latent_list=True)
                            else:
                                f_ranker = None
                            idx_list = list(range(idx - opt['network_STDR']['radius'], idx + opt['network_STDR']['radius'] + 1))
                            idx_list = np.clip(idx_list, 0, nfs - 1)
                            input_data = []
                            for idx_ in idx_list:
                                input_data.append(x[:, 0, idx_])
                            input_data = torch.stack(input_data, 1)
                            result_data = restoration_model(input_data.contiguous(), qp=args.qp, f_ranker=f_ranker)
                            output_data.append(result_data)
                        output_data = torch.stack(output_data, dim=2)
                        output_data = torch.cat([output_data, x[:, 1:]], 1)
                        v_seg = rearrange(output_data, 'b c t h w -> b t c h w')[0]

                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        I0, I2 = group_list[i], group_list[i + 1]
                        xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(device, non_blocking=True)
                        batch_size = xs.shape[0]
                        s_shape = xs.shape[-2:]
                        time_step = 2
                        coord_inputs = [
                            (model.sample_coord_input(batch_size, s_shape, [(j + 1) * (1.0 / time_step)], device=xs.device), None)
                            for j in range(time_step - 1)
                        ]
                        t = [
                            (i + 1) * (1.0 / time_step) * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                            for i in range(time_step - 1)]
                        with torch.no_grad():
                            all_outputs = model(xs, coord_inputs, t=t)
                        pred = all_outputs["imgt_pred"][0]
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = v_seg.cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'VideoINR':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred = single_forward(model, torch.stack([group_list[i].cuda(device), group_list[i + 1].cuda(device)], dim=1).cuda(device), 1, 3)
                        pred = pred[1]
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = v_seg.cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'MoMo':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred, _ = model(torch.stack([group_list[i], group_list[i + 1]], dim=2).cuda(device), num_inference_steps=20, resize_to_fit=True, pad_to_fit_unet=False)
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = v_seg.cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name in ['CVRS']:
                    group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    padder = InputPadder(group_1.shape, divisor=64)
                    group_1_pad = padder.pad(group_1)[0]

                    down_size = (4, group_1_pad.shape[-2], group_1_pad.shape[-1])
                    x_down = rescale_model.inference_down(rearrange(ycbcr2rgb(group_1_pad), 't c h w -> c t h w').unsqueeze(0), down_size)
                    LR_img = model.inference_latent2RGB(x_down)
                    LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> b t c h w')[0]).unsqueeze(0), 'b t c h w -> b c t h w')

                    Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    v_seg = v_seg[:, [0, 2, 1]]
                    img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]

                    rev_back = model.inference_RGB2latent(rearrange(ycbcr2rgb(v_seg).unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'))
                    out_x = rescale_model.inference_up(rev_back, (group_1_pad.shape[0], group_1_pad.shape[-2], group_1_pad.shape[-1]))
                    out_x = padder.unpad(out_x)
                    out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
                    out_x = rgb2ycbcr(out_x)
                    pred = out_x[1::2]

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
                    LFR_lq = padder.unpad(LFR_lq)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name in ['CVRS_finetuned']:
                    group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    padder = InputPadder(group_1.shape, divisor=64)
                    group_1_pad = padder.pad(group_1)[0]

                    down_size = (4, group_1_pad.shape[-2], group_1_pad.shape[-1])
                    LF = model.test_long(input=group_1_pad.unsqueeze(0), rev=False)
                    b = LF.shape[0]
                    LF = rearrange((rearrange(LF, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b)

                    Quantization_H265_Stream.open_writer('cpu', LF.shape[-1], LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name + str(args.slice_id), mode=args.mode)
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF[:, [0, 2, 1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    v_seg = v_seg[:, [0, 2, 1]]
                    img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]

                    out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), rev=True)
                    out_x = padder.unpad(out_x)
                    out_x = rearrange(out_x, 'b t c h w -> b t c h w')[0]
                    pred = out_x[1::2]

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
                    LFR_lq = padder.unpad(LFR_lq)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                elif model_name == 'RIFE':
                    Quantization_H265_Stream.open_writer('cpu', img1.shape[-2], img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
                    LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
                    out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
                    gt = rearrange(torch.stack([img2, img4, img6], dim=0), 'b h w c -> b c h w')
                    Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
                    _, img_distri = Quantization_H265_Stream.close_writer()

                    Quantization_H265_Stream.open_reader(verbosity=0)
                    v_seg = Quantization_H265_Stream.read_multi_frames(4)
                    img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
                    padder = InputPadder(img1_lq.shape, divisor=32)
                    img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
                    group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
                    pred_list = []
                    out_list = []

                    for i in range(3):
                        pred = model.inference(group_list[i], group_list[i + 1])
                        pred = padder.unpad(pred)
                        out_list.append(padder.unpad(group_list[i]))
                        out_list.append(pred)
                        pred_list.append(pred)
                    out_list.append(padder.unpad(group_list[-1]))
                    pred = torch.cat(pred_list, dim=0)
                    out_x = torch.cat(out_list, dim=0)

                    LFR_gt = rearrange(torch.stack([img1, img3, img5, img7], dim=0), 'b h w c -> b c h w')
                    LFR_lq = v_seg.cuda(device)
                    LFR_hq = out_x[[0, 2, 4, 6]]

                else:
                    raise Exception('invalid model name')

                out_rgb = ycbcr2rgb(out_x)
                out_label_rgb = ycbcr2rgb(out_label)
                gt_rgb = ycbcr2rgb(gt)
                pred_rgb = ycbcr2rgb(pred)

                ssim = ssim_matlab(out_label_rgb, torch.round(out_rgb * 255) / 255.).detach().cpu().numpy()
                ssim_inter = ssim_matlab(gt_rgb, torch.round(pred_rgb * 255) / 255.).detach().cpu().numpy()

                out_label_rgb = out_label_rgb.cuda(device)
                out_rgb = out_rgb.cuda(device)
                gt_rgb = gt_rgb.cuda(device)
                pred_rgb = pred_rgb.cuda(device)

                FID_list.append(0)

                out_x = (np.round(out_x.cpu().numpy() * 255) / 255.).clip(min=0, max=1)
                out_label = out_label.cpu().numpy()

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

                bpp_list.append(img_distri)

                if model_name != 'codec_reference':
                    vmaf_score = calculate_vmaf_score(out_label, out_x, out_x.shape[-1], out_x.shape[-2], pix_fmt='yuv444p')
                    if vmaf_score is not None:
                        vmaf_list.append(vmaf_score)

                    gt_seq = out_label_rgb.unsqueeze(0).clamp(0, 1)
                    rec_seq = out_rgb.unsqueeze(0).clamp(0, 1)
                    B, T, C, H, W = gt_seq.shape

                    gt_flat = gt_seq.view(-1, C, H, W)
                    rec_flat = rec_seq.view(-1, C, H, W)
                    lpips_vals_flat = lpips(gt_flat, rec_flat).repeat(gt_flat.shape[0])
                    dists_vals_flat = dists(gt_flat, rec_flat).squeeze()
                    lpips_vals = lpips_vals_flat.view(B, T)
                    dists_vals = dists_vals_flat.view(B, T)

                    curr_gt = gt_seq[:, 1:, :, :, :]
                    prev_gt = gt_seq[:, :-1, :, :, :]
                    curr_rec = rec_seq[:, 1:, :, :, :]
                    prev_rec = rec_seq[:, :-1, :, :, :]

                    flow_rec = get_flow(of_model, rearrange(curr_rec, 'b t c h w -> (b t) c h w'), rearrange(prev_rec, 'b t c h w -> (b t) c h w'))
                    flow_gt = get_flow(of_model, rearrange(curr_gt, 'b t c h w -> (b t) c h w'), rearrange(prev_gt, 'b t c h w -> (b t) c h w'))
                    flow_rec = rearrange(flow_rec, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])
                    flow_gt = rearrange(flow_gt, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])

                    warped_rec = flow_warp(rearrange(prev_rec, 'b t c h w -> (b t) c h w'), flow_rec)
                    warped_rec = rearrange(warped_rec, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])

                    mse = ((warped_rec - curr_gt) ** 2).flatten(2).mean(dim=2)
                    w_psnr_vals = -10 * torch.log10(mse + 1e-8)

                    c_gt_flat = curr_gt.reshape(-1, C, H, W)
                    p_gt_flat = prev_gt.reshape(-1, C, H, W)
                    c_rec_flat = curr_rec.reshape(-1, C, H, W)
                    p_rec_flat = prev_rec.reshape(-1, C, H, W)
                    lpips_gt_t = lpips(c_gt_flat, p_gt_flat).squeeze()
                    lpips_rec_t = lpips(c_rec_flat, p_rec_flat).squeeze()
                    tlpips_vals_flat = (lpips_gt_t - lpips_rec_t).abs()
                    tlpips_vals = tlpips_vals_flat.repeat(B, T - 1)

                    tof_vals = (flow_rec - flow_gt).abs().flatten(2).mean(dim=2)

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
                    print(f"psnr:{np.mean(psnr_list):.2f},psnr_avg_mse:{(np.mean(mse_list)):.10f},ssim:{np.mean(ssim_list):.4f},psnr_LFR_lq:{np.mean(LFR_lq_psnr_list):.2f},ssim_LFR_lq:{np.mean(LFR_lq_ssim_list):.4f},psnr_LFR_hq:{np.mean(LFR_hq_psnr_list):.2f},sigma:{np.mean(sigma_list):.4f},psnr inter:{np.mean(psnr_inter_list):.2f},ssim inter:{np.mean(ssim_inter_list):.4f},lpips:{mean_lpips:.4f},dists:{mean_dists:.4f},tlpips(1e3):{mean_tlpips:.2f},tof(1e1):{mean_tof:.4f},warpping_psnr:{mean_warpping_psnr:.2f},vmaf:{np.mean(vmaf_list):.4f},ave_img_bpp:{np.mean(bpp_list):.6f}")