import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
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
from benchmark.utils.padder import InputPadder
import torch.nn.functional as F
from collections import OrderedDict
import options.options as option
from models.VRN_model import TVRNCodecModel as Model
from PIL import Image
from EBME.bi_flownet import BiFlowNet
from EBME.fusionnet import FusionNet
from models.VRN_model import STAAModel
from models.modules.STDR_Net import Net as STDR_Net
import torch.nn as nn
from XVFI.XVFInet import XVFInet
import yaml
from torch.nn.parallel import DataParallel, DistributedDataParallel

'''==========import from our code=========='''
sys.path.append('.')
from benchmark.utils.pytorch_msssim import ssim_matlab
from utils.functions import ycbcr2rgb, rgb2ycbcr




from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from DISTS_pytorch import DISTS
# from torchvision.models.optical_flow import raft_large as raft
from torchvision.transforms import ToTensor
# import sys
# sys.path.append('/code/codes/RAFT/core')
from RAFT.core.raft import RAFT as raft


import tempfile
import subprocess
import json
vmaf_root = '/tmp/vmaf-1.3.15'
os.environ['PYTHONPATH'] = f"{vmaf_root}/libsvm/python:{vmaf_root}/python/src:{vmaf_root}:" + os.environ.get('PYTHONPATH', '')
def calculate_vmaf_score(ref_np, dist_np, width, height, pix_fmt='yuv444p'):
    """
    将 Tensor 保存为临时 YUV 文件并调用 run_vmaf.py 计算分数
    ref_tensor, dist_tensor: shape (T, C, H, W) 或 (B, T, C, H, W), 范围 [0, 1], YUV 格式
    """    
    # # 确保是 CPU numpy 数组
    # if len(ref_tensor.shape) == 5: # (B, T, C, H, W) -> 取第一个 batch
    #     ref_tensor = ref_tensor[0]
    #     dist_tensor = dist_tensor[0]
    
    # ref_np = ref_tensor.cpu().numpy()
    # dist_np = dist_tensor.cpu().numpy()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f_ref, \
         tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f_dist:
        ref_path = f_ref.name
        dist_path = f_dist.name
        
        # 转换并写入 YUV (假设输入是 YUV 格式，如果是 RGB 需要先转 YUV)
        # 注意：你的脚本中 out_label 和 out_x 已经是 YUV 格式 (见代码 ycbcr2rgb/rgb2ycbcr 的使用)
        # 这里假设 Tensor 是 (T, C, H, W), 值域 [0, 1]
        def write_yuv(tensor, path, w, h, fmt):
            T, C, H, W = tensor.shape
            with open(path, 'wb') as f:
                for t in range(T):
                    frame = (tensor[t] * 255.0).clip(0, 255).astype(np.uint8)
                    if fmt == 'yuv444p':
                        # Y, U, V 平面拼接
                        f.write(frame[0].tobytes()) # Y
                        f.write(frame[1].tobytes()) # U
                        f.write(frame[2].tobytes()) # V
                    elif fmt == 'yuv420p':
                        # 需要下采样 U/V，这里简化处理，假设输入已是 420 或直接用 444 跑分更准
                        # 为了简单，如果原图是 444，建议直接用 yuv444p 跑分
                        f.write(frame[0].tobytes())
                        f.write(frame[1].tobytes())
                        f.write(frame[2].tobytes())
        
        write_yuv(ref_np, ref_path, width, height, pix_fmt)
        write_yuv(dist_np, dist_path, width, height, pix_fmt)

    # 构建命令
    # 注意：模型路径需要根据你的实际情况修改，如果不用特定模型可去掉 --model 参数
    cmd = [
        'python3', f'{vmaf_root}/python/script/run_vmaf.py',
        pix_fmt, str(width), str(height),
        ref_path, dist_path,
        '--model /tmp/vmaf-1.3.15/model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl',
        '--out-fmt', 'json',
        # '--ci' # 如果需要置信区间
    ]
    
    # 如果有自定义模型，取消下面这行的注释并修改路径
    # cmd.extend(['--model', '/path/to/your/model.pkl'])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_json = json.loads(result.stdout)
        
        # 提取分数
        # 根据之前的输出，分数可能在 'aggregate' -> 'VMAF_score' 或 'BOOTSTRAP_VMAF_score'
        aggregate = output_json.get('aggregate', {})
        score = aggregate.get('VMAF_score') or aggregate.get('BOOTSTRAP_VMAF_score')
        
        return score
    except Exception as e:
        print(f"VMAF calculation failed: {e}")
        print(f"Stderr: {result.stderr if 'result' in locals() else 'N/A'}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(ref_path): os.remove(ref_path)
        if os.path.exists(dist_path): os.remove(dist_path)




import gc

def clear_gpu_memory(device_id=0):
    """
    彻底清理指定 GPU 的显存
    """
    # 1. 强制垃圾回收 (Python 层面)
    # 清理那些引用计数为 0 但尚未被释放的对象
    gc.collect()
    
    # 2. 清空 PyTorch 缓存分配器
    # 这会将未使用的显存归还给操作系统，而不仅仅是标记为“可用”
    torch.cuda.empty_cache()
    
    # 3. (可选) 重置累积的内存统计信息
    # 如果你在做性能分析，这很有用，否则可以跳过
    torch.cuda.reset_peak_memory_stats(device_id)
    
    # 4. 打印当前状态以确认
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        print(f"✅ GPU {device_id} 清理完成:")
        print(f"   已分配显存: {allocated:.2f} MB")
        print(f"   预留显存: {reserved:.2f} MB")
    else:
        print("⚠️ CUDA 不可用，无法清理 GPU 显存。")


def get_flow(of_model, target, source, rescale_factor=1):
    flows = of_model(target, source)
    flow = flows[-1]
    flow = F.interpolate(flow//rescale_factor, scale_factor=1/rescale_factor, mode='bilinear') if rescale_factor != 1 else flow
    flow = flow.permute(0, 2, 3, 1) # permute to B, H, W, 2
    return flow


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
            
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
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

    B = flow.shape[0]
    H = flow.shape[1]
    W = flow.shape[2]

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
    # inputs: flow_forward, flow_backward
    # return: occlusion mask 
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    tmp = bw_flow # to this for divergence between their and my interpretation of forward and backward of
    bw_flow = fw_flow
    fw_flow = tmp
    
    fw_flow_w = flow_warp(fw_flow.permute(0,3,1,2), bw_flow).permute(0,2,3,1)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.ones((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2])).to(device)
    occlusion[mask == 1] = 0

    return occlusion

def warp_error(of_model, current_frame, prev_frame, current_gt, prev_gt, use_occlusion_mask=True):
    flow_forward, flow_backward = get_flow_forward_backward(of_model, current_gt, prev_gt)
    prev_warped = flow_warp(prev_frame, flow_forward)
    prev_gt_warped = flow_warp(prev_gt, flow_forward)
    if use_occlusion_mask:
        mask = detect_occlusion(flow_forward, flow_backward)
        valid_pixels = torch.sum(mask == 1)
        mean_error = torch.sum((mask*current_frame - mask * prev_warped)**2) / (valid_pixels*3+1e-10)
    else:
        mean_error = ((current_frame - prev_warped)**2).mean()
    return mean_error


from torch.utils.data import Dataset, DataLoader


test_file = ''

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
# parser.add_argument('--window_size', type=int, default=2, help='window size')
# parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
parser.add_argument('--checkpoints', type=str, default=None)
parser.add_argument('-staa_opt', type=str, help='Path to option YMAL file.')

parser.add_argument('--codec_type', type=str, default='hevc', choices=['avc', 'hevc', 'av1', 'vp9', 'vvc'])
parser.add_argument('--slice_id', type=int, default=0, help='当前任务的切片索引 (0 to total_slices-1)')
parser.add_argument('--total_slices', type=int, default=1, help='总切片数量')
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的CUDA设备ID')
parser.add_argument('--small', default=False, help='use small model')
parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
parser.add_argument('--alternate_corr', default=False, help='use efficent correlation implementation')
parser.add_argument('--dataset_type', type=str, default='septuplet', choices=['septuplet', '65frames'], help='use small model')

args = parser.parse_args()

if args.model == 'TVRN_S':
    args.opt = '/code/codes/options/test_septuplet/test_TVRN_without_restoration.yml'
    args.checkpoints = '/model/fengxm/VRN/MIMO_VRN/checkpoints/model_wo_restoration.pth'

opt = option.parse(args.opt, is_train=False)
opt['codec_type'] = args.codec_type

if args.checkpoints is not None:
    opt['path']['pretrain_model_G'] = args.checkpoints
    print('loading weight from ', opt['path']['pretrain_model_G'])
# opt['network_G']['entropy_model'] = True

# # vfips
# moduleNetwork = networks.get_model('multiscale_v33', depth_ksize=args.depth_ksize,opt=args)
# moduleNetwork.load_state_dict(torch.load(args.expdir + 'model.pytorch'), strict=True)
# moduleNetwork.eval()


# 1. 设置全局默认 CUDA 设备 (返回 None，不要赋值给 device 变量)
torch.cuda.set_device(args.gpu_id)

# 2. 创建 device 对象 (此时可以不带 ID，因为它会使用默认值，但带上更明确)
device = torch.device(f"cuda:{args.gpu_id}")


import shutil
BASE_OUTPUT_DIR = "/data/fengxm/vimeo90k/tvrn_revision"


model_name = args.model
# assert model_name in ['EMA', 'TVRN', 'SGM', 'UPR', 'UPR_l', 'UPR_L', 'IFRNet']

'''==========Model setting=========='''
TTA = True
average_metric = True
if 'TVRN' in model_name:
    model = Model(opt)
elif model_name == 'STAA':
    staa_opt = option.parse(args.staa_opt, is_train=False)
    model = STAAModel(staa_opt, device)
elif model_name == 'EMA':
    # args.model = 'ours'
    import config as cfg
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
    from Trainer import Model
    model = Model(-1)
    model.load_model(path = opt['path']['EMA_model'])
    model.eval()
    model.device()
elif model_name == 'SGM':
    import config_SGM as cfg
    from models_SGM.Trainer_x4k import Model
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
    from models.VFI_model import UPRModelBase
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
    model.cuda(device)
elif model_name == 'UPR_l':
    from models.VFI_model import UPRModelLarge
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
    model.cuda(device)
elif model_name == 'UPR_L':
    from models.VFI_model import UPRModelLLarge
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
    model.cuda(device)
elif model_name == 'IFRNet':
    from models.IFRNet_L import Model
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
    model.cuda(device)
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
    from models_VFIformer.modules import define_G
    # python FILM_test.py --data_root [your SNU-FILM path] 
    # --test_level [easy/medium/hard/extreme] --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth
    args.test_level = 'medium'
    args.net_name = 'VFIformer'
    args.resume = '/data/fengxm/vimeo90k/pretrained_model/VFIformer/net_220.pth'
    args.dist = False
    args.gpu_ids = [0, ]
    # args.device = device
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
    bi_flownet_args.model_file =  '/data/fengxm/vimeo90k/pretrained_model/EBME/ebme/bi-flownet.pkl'

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
    model = model.cuda(device)
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

    if model_name == 'GIMM+VQE':
        restoration_model = STDR_Net(opt['network_STDR'], type='STDR_both_adaptor')
        # 加载模型权重
        load_path_G = opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('Loading restoration model for G [{:s}] ...'.format(load_path_G))
            load_restoration(load_path_G, restoration_model, True)
        # ranker
        from models.modules.Inv_arch import Ranker_wo_res
        ranker_inchans = 3
        ranker_outchans=1
        ranker = Ranker_wo_res(in_chans=ranker_inchans, out_chans=ranker_outchans)
        load_path_ranker = opt['path']['pretrain_ranker']
        if load_path_ranker is not None:
            load_ranker(load_path_ranker, ranker, True)
elif model_name == 'VideoINR':
    import models_videoINR.models.modules.Sakuya_arch as Sakuya_arch
    # device = device
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
    model.cuda(device)
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
    # param = get_model_total_params(rescale_model)
    # print(f'param: {param}')

elif model_name == 'CVRS_finetuned':
    # 使用所提出的代理网络对CVRS进行finetune

    from models.VRN_model import CSTVRModel as Model
    opt['path']['pretrain_model_G'] = '/model/fengxm/VRN/MIMO_VRN/CSTVR_w_surrogate_net/70000_G.pth'
    model = Model(opt)

elif model_name == 'RIFE':
    from models_RIFE.train_log.RIFE_HDv3  import Model
    model = Model(device=device)
    model.load_model('/code/codes/models_RIFE/train_log', -1)
    model.eval()
    model.device()


print(f'=========================Starting testing=========================')
print(f'Dataset: UCF101   Model: {model_name}   TTA: {TTA}')
path = args.path
# dirs = os.listdir(path)
# level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
# level_list = ['sep_testlist.txt',] 
Quantization_H265_Stream = Quantization_H265_Stream(args.qp,-1,None,opt)

psnr_list, ssim_list = [], []
mse_list = []
sigma_list = []
psnr_lr_list, ssim_lr_list = [], []
psnr_inter_list, ssim_inter_list = [], []
floLPIPS_list, VFIPS_list = [], []
LPIPS_list, DISTS_list, FID_list = [], [], []
file_list = []
bpp_list = []
psnr_LFR = []
LFR_lq_psnr_list, LFR_lq_ssim_list, LFR_hq_psnr_list = [], [], []
vmaf_list = []
file_list = os.listdir(path)
LFR_gt, LFR_hq, LFR_lq = None, None, None

# file_list = file_list[:80]

if args.total_slices > 1:
    total_len = len(file_list)
    slice_size = (total_len + args.total_slices - 1) // args.total_slices  # 向上取整
    start_idx = args.slice_id * slice_size
    end_idx = min(start_idx + slice_size, total_len)
    
    file_list = file_list[start_idx:end_idx]
    print(f"[GPU {args.gpu_id}] Processing slice {args.slice_id}/{args.total_slices}: items {start_idx} to {end_idx-1}")
else:
    print(f"[GPU {args.gpu_id}] Processing full list.")


# 输出图片路径
output_folder = os.path.join(
    BASE_OUTPUT_DIR, 
    args.codec_type,      # 例如: hevc, av1
    'ucf101',
    args.model,           # 例如: TVRN, EMA
    f"QP{args.qp}"        # 例如: QP28
)
# if os.path.exists(output_folder) and args.slice_id == 0:
#     print(f"Cleaning existing folder: {output_folder}")
#     shutil.rmtree(output_folder)
# output_folder_in = os.path.join(output_folder, 'input')
output_folder_in = args.path
output_folder_out = os.path.join(output_folder, 'output')

# os.makedirs(output_folder_in, exist_ok=True)
# os.makedirs(output_folder_out, exist_ok=True)
# print(f"Created output folder: {output_folder}")



# def run_evaluation_sequence_batch(output_folder_in, output_folder_out, args, device, batch_size=4, num_workers=4, seq_length=7):
# 1. 初始化模型
of_model = raft(args)
ckpt_path = '/data/fengxm/vimeo90k/tvrn_revision/raft_ckpts/raft-things.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
new_ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
of_model.load_state_dict(new_ckpt)
of_model.to(device)
of_model.eval()

lpips = LPIPS(reduction='mean').to(device)
dists = DISTS().to(device)

# 结果容器
lpips_dict = {}
dists_dict = {}
tlpips_dict = {}
tof_dict = {}
warp_psnr_dict = {}
w_psnr_dict = {}



# file_list = file_list[:10]

# for line in (file_list):
for line_id, line in (enumerate(file_list)):
    # if line_id > 5:
    #     break
    # if '/data/fengxm/vimeo90k/snufilm_test/test/GOPRO_test/GOPR0384_11_05' not in line[0]:
    #     continue 
    I0_path = os.path.join(path,   line, 'img_0.png')
    I1_path = os.path.join(path,   line, 'img_1.png')
    I2_path = os.path.join(path,   line, 'img_2.png')
    I3_path = os.path.join(path,   line, 'img_3.png')
    I4_path = os.path.join(path,   line, 'img_4.png')
    I5_path = os.path.join(path,   line, 'img_5.png')
    I6_path = os.path.join(path,   line, 'img_6.png')

    if 'TVRN' in model_name:
        # img1 = torch.tensor(cv2.cvtColor(cv2.imread(I0_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.
        # img2 = torch.tensor(cv2.cvtColor(cv2.imread(I1_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.
        # img3 = torch.tensor(cv2.cvtColor(cv2.imread(I2_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.            

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
        padder = InputPadder(group_1.shape, divisor=64)
        group_1_pad = padder.pad(group_1)[0]
        LF, HF = model.test_long(input=group_1_pad.unsqueeze(0), qp=args.qp, rev=False)  # b t c h w  -> b c t h w
        # LF = padder.unpad(LF)
        # HF = padder.unpad(HF)
        Quantization_H265_Stream.open_writer('cpu',LF.shape[-1],LF.shape[-2], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)
        
        # # write in frame-by-frame
        # Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,:,0:1], 'b c t h w -> b (t c) h w').detach().to("cpu"))
        # Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,:,1:2], 'b c t h w -> b (t c) h w').detach().to("cpu"))
        
        # Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        Quantization_H265_Stream.write_multi_frames(rearrange(LF[:,[0,2,1]], 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        outsouts2 = []
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        v_seg = v_seg[:,[0,2,1]]
        img1_recon, img3_recon, img5_recon, img7_recon = v_seg[0], v_seg[1], v_seg[2], v_seg[3]
        # v_seg_pad = padder.pad(v_seg)[0]
        out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
        out_x = padder.unpad(out_x)
        # b c t h w
        out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
        pred = out_x[1:2]
        
        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
        LFR_lq = padder.unpad(LFR_lq)
        LFR_hq = out_x[[0,2,4,6]]

    elif model_name == 'EMA':
        # img1 = torch.tensor(cv2.cvtColor(cv2.imread(I0_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.
        # img2 = torch.tensor(cv2.cvtColor(cv2.imread(I1_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.
        # img3 = torch.tensor(cv2.cvtColor(cv2.imread(I2_path), cv2.COLOR_BGR2YUV)).float().cuda(device) / 255.

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        
        # # write in frame-by-frame
        # Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        # LF = rearrange((img1).unsqueeze(0).unsqueeze(0), 'b t h w c -> b c t h w')
        # Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        # LF = rearrange((img3).unsqueeze(0).unsqueeze(0), 'b t h w c -> b c t h w')
        # Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))

        
        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
            pred = padder.unpad(pred)
            # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred.unsqueeze(0))
            pred_list.append(pred.unsqueeze(0))
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)


        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = v_seg.cuda(device)
        LFR_hq = out_x[[0,2,4,6]]
        
        # 可视化
        # import matplotlib.pyplot as plt
        # plt.imsave('/code/demo.png', cv2.cvtColor((rearrange(pred[0], 'c h w -> h w c').cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_YUV2RGB))
    elif model_name == 'SGM':
        
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
            pred = model.hr_inference(group_list[i], group_list[i+1], TTA=TTA, down_scale=0.5, fast_TTA=False).clamp(0.0, 1.0)
            pred = padder.unpad(pred)
            # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)



    elif model_name == 'RIFE':

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            pred = model.inference(group_list[i], group_list[i+1])
            pred = padder.unpad(pred)
            # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)


        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = v_seg.cuda(device)
        LFR_hq = out_x[[0,2,4,6]]
        
        
    elif model_name in ['UPR', 'UPR_l', 'UPR_L']:

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
            # pred = model.hr_inference(group_list[i], group_list[i+1], TTA=TTA, down_scale=2, fast_TTA=False).clamp(0.0, 1.0)
            pred, _, _ = model(group_list[i], group_list[i+1], time_period = 0.5)
            pred = padder.unpad(pred)
            # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)
        
    elif model_name in ['IFRNet']:

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda(device)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=20)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
            pred = model.inference(group_list[i], group_list[i+1], embt, scale_factor=0.8)
            pred = padder.unpad(pred)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        out_x = torch.cat(out_list, dim=0)
        pred = torch.cat(pred_list, dim=0)

    elif model_name == 'STAA':
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
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
            recon_3_frames = model.netG(x=[torch.stack([group_list[2-i].cuda(device), group_list[3-i].cuda(device)], dim=1), None], rev=True)
            out_list.append(padder.unpad(recon_3_frames[:,-1]))
            out_list.append(padder.unpad(recon_3_frames[:,-2]))
            pred_list.append(padder.unpad(recon_3_frames[:,-2]))
            group_list[2-i] = recon_3_frames[:,-3]
        out_list.append(padder.unpad(recon_3_frames[:,-3]))
        out_x = torch.cat(out_list[::-1], dim=0)  # b t c h w
        pred = torch.cat(pred_list[::-1], dim=0)

        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = torch.cat([img1_lq, img2_lq, img3_lq, img4_lq], dim=0).cuda(device)
        LFR_hq = out_x[[0,2,4,6]]


    elif model_name in ['codec_reference']:

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(7)
        out_x = torch.tensor(v_seg).cuda(device)
        pred = out_x[1::2]

    elif model_name == 'VFIformer':

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=64)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # pred = model.inference(group_list[i], group_list[i+1], embt=embt, scale_factor=0.8)
            # inference process
            # pred = model.inference(img1_lq_pad2, img2_lq_pad2, embt=embt, scale_factor=0.8)
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
            flow_down = model.get_flow(img1_down.cuda(device), img3_down.cuda(device))
            if h % 64 != 0 or w % 64 != 0:
                flow_down = flow_down[:, :, :h, :w]
            flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale
            pred, _,  = model(img1_lq_pad.cuda(device), img2_lq_pad.cuda(device), flow_pre=flow)
            
            pred = padder.unpad(pred)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)
        
    elif model_name == 'EBME':
        
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
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
                    
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
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
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # model inference
            I0, I2 = group_list[i], group_list[i+1]
            xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
                device, non_blocking=True
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
            with torch.no_grad():
                all_outputs = model(
                    xs,
                    coord_inputs,
                    t=[
                        (i + 1)
                        * (1.0 / time_step)
                        * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                        for i in range(time_step - 1)
                    ],
                )
            # all_outputs = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            pred = all_outputs["imgt_pred"][0]
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(padder.unpad(pred))
            pred_list.append(padder.unpad(pred))
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)

        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = v_seg.cuda(device)
        LFR_hq = out_x[[0,2,4,6]]
    elif model_name == 'VideoINR':
        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # I0, I2 = group_list[i], group_list[i+1]
            # xs = torch.stack((I0, I2), dim=1).to(device, non_blocking=True)  # 2,c,h,w
            pred = single_forward(model, torch.stack([group_list[i].cuda(device), group_list[i+1].cuda(device)], dim=1).cuda(device), 1, 3)
            pred = pred[1]
            pred = padder.unpad(pred)
            # out_x = torch.cat([img1_lq, pred.unsqueeze(0), img2_lq], dim=0)
            out_list.append(padder.unpad(group_list[i]))
            out_list.append(pred)
            pred_list.append(pred)
        out_list.append(padder.unpad(group_list[-1]))
        pred = torch.cat(pred_list, dim=0)
        out_x = torch.cat(out_list, dim=0)

        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = v_seg.cuda(device)
        LFR_hq = out_x[[0,2,4,6]]
    elif model_name == 'MoMo':

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        Quantization_H265_Stream.open_writer('cpu',img1.shape[-2],img1.shape[-3], pix_fmt='yuv444p', verbosity=0, extra_info=model_name, mode=args.mode)  # w,, h
        LF = rearrange(torch.stack([img1, img3, img5, img7], dim=0).unsqueeze(0), 'b t h w c -> b c t h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([img2, img4, img6, ], dim=0), 'b h w c -> b c h w')
        Quantization_H265_Stream.write_multi_frames(rearrange(LF, 'b c t h w -> b (t c) h w').detach().to("cpu"))
        _,img_distri = Quantization_H265_Stream.close_writer()
        Quantization_H265_Stream.open_reader(verbosity=0)
        v_seg = Quantization_H265_Stream.read_multi_frames(4)
        img1_lq, img2_lq, img3_lq, img4_lq = v_seg[0:1].cuda(device), v_seg[1:2].cuda(device), v_seg[2:3].cuda(device), v_seg[3:4].cuda(device)
        padder = InputPadder(img1_lq.shape, divisor=32)
        img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad = padder.pad(img1_lq, img2_lq, img3_lq, img4_lq)
        group_list = [img1_lq_pad, img2_lq_pad, img3_lq_pad, img4_lq_pad]
        pred_list = []
        out_list = []
        for i in range(3):
            # pred = model.inference(group_list[i], group_list[i+1], TTA=TTA, fast_TTA=TTA)[0]
            pred, _ = model(
                torch.stack([group_list[i], group_list[i+1]], dim=2).cuda(device),
                num_inference_steps=8,
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
        LFR_lq = v_seg.cuda(device)
        LFR_hq = out_x[[0,2,4,6]]

    elif model_name == 'CVRS':

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

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

        rev_back = model.inference_RGB2latent(rearrange(ycbcr2rgb(v_seg).unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'))
        out_x = rescale_model.inference_up(rev_back, (group_1_pad.shape[0], group_1_pad.shape[-2], group_1_pad.shape[-1]))
        # out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
        out_x = padder.unpad(out_x)
        # b c t h w
        out_x = rearrange(out_x, 'b c t h w -> b t c h w')[0]
        out_x = rgb2ycbcr(out_x)
        pred = out_x[1::2]
        # pred = rgb2ycbcr(pred)

        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
        LFR_lq = padder.unpad(LFR_lq)
        LFR_hq = out_x[[0,2,4,6]]



    elif model_name in ['CVRS_finetuned']:

        img1 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I0_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img2 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I1_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img3 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I2_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img4 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I3_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img5 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I4_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img6 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I5_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')
        img7 = rearrange(rgb2ycbcr(torch.tensor(rearrange((cv2.imread(I6_path)[:,:,[2,1,0]].astype(np.float) / 255.), 'h w c -> c h w')).float().cuda(device)), 'c h w -> h w c')

        group_1 = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        out_label = rearrange(torch.stack([img1, img2, img3, img4, img5, img6, img7], dim=0), 't h w c -> t c h w')
        gt = rearrange(torch.stack([ img2, img4, img6 ], dim=0), 'b h w c -> b c h w')
        padder = InputPadder(group_1.shape, divisor=64)
        group_1_pad = padder.pad(group_1)[0]

        down_size = (4, group_1_pad.shape[-2] , group_1_pad.shape[-1])

        LF = model.test_long(input=group_1_pad.unsqueeze(0), rev=False)  # b t c h w  -> b c t h w
        b = LF.shape[0]
        LF = rearrange((rearrange(LF, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b)

        # x_down = rescale_model(rearrange(ycbcr2rgb(group_1_pad), 't c h w -> c t h w').unsqueeze(0), down_size, rev=False)
        # LR_img = model(x_down, latent_to_RGB=True)
        # LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> b t c h w')[0]).unsqueeze(0), 'b t c h w -> b c t h w')

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

        # rev_back = model(rearrange(ycbcr2rgb(v_seg).unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), latent_to_RGB=False)
        # out_x = rescale_model(rev_back, (group_1_pad.shape[0], group_1_pad.shape[-2], group_1_pad.shape[-1]), rev=True)
        # out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), qp=args.qp, rev=True, saved_HF=None)
        out_x = model.test_long(input=rearrange(v_seg.unsqueeze(0).cuda(device), 'b t c h w -> b c t h w'), rev=True)

        out_x = padder.unpad(out_x)
        # b c t h w
        out_x = rearrange(out_x, 'b t c h w -> b t c h w')[0]
        # out_x = rgb2ycbcr(out_x)
        pred = out_x[1::2]
        # pred = rgb2ycbcr(pred)

        LFR_gt = rearrange(torch.stack([ img1, img3, img5, img7 ], dim=0), 'b h w c -> b c h w')
        LFR_lq = torch.stack([img1_recon, img3_recon, img5_recon, img7_recon], dim=0).cuda(device)
        LFR_lq = padder.unpad(LFR_lq)
        LFR_hq = out_x[[0,2,4,6]]


    else:
        raise Exception('invalid model name')
  
    if model_name == 'x265_latency':
        out_rgb = ycbcr2rgb(out_x)
        out_label_rgb = ycbcr2rgb(out_label)
        ssim = ssim_matlab(out_label_rgb, torch.round(out_rgb * 255) / 255.).detach().cpu().numpy()
        out_label_rgb = out_label_rgb.cuda(device)      
        out_rgb = out_rgb.cuda(device)
        out_x = (np.round(out_x.cpu().numpy() * 255) / 255.).clip(min=0, max=1)
        out_label = out_label.cpu().numpy()
        psnr = -10 * math.log10(((out_label - out_x) * (out_label - out_x)).mean())
        mse = ((out_label - out_x) * (out_label - out_x)).mean()
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)
        bpp_list.append(img_distri)
        continue

    # out_x = out_x[:,  :, :210, :255]   
    # out_label = out_label[:, :, :210, :255]  
    # gt = gt[:, :, :210, :255]
    # pred = pred[:,  :, :210, :255]

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
    # average mse
    psnr = -10 * math.log10(((out_label - out_x) * (out_label - out_x)).mean())

    mse = ((out_label - out_x) * (out_label - out_x)).mean()
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    mse_list.append(mse)
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

    bpp_list.append(img_distri)
    # print(psnr)
    pass
    
    if model_name != 'codec_reference':

        T, C, H, W = out_label.shape
        vmaf_score = calculate_vmaf_score(out_label, out_x, W, H, pix_fmt='yuv444p')
        vmaf_list.append(vmaf_score)
        # if vmaf_score is not None:
        #     print(f"Sequence {line}: VMAF Score = {vmaf_score:.4f}")

        # 数据形状: (T, C, H, W) -> (B, T, C, H, W)
        gt_seq = out_label_rgb.unsqueeze(0).clamp(0,1)
        rec_seq = out_rgb.unsqueeze(0).clamp(0,1)
        
        B, T, C, H, W = gt_seq.shape
        
        # --- A. 逐帧指标 (Frame-wise) ---
        # 展平 B 和 T 以便逐帧计算 LPIPS/DISTS
        # View: (B, T, C, H, W) -> (B*T, C, H, W)
        gt_flat = gt_seq.view(-1, C, H, W)
        rec_flat = rec_seq.view(-1, C, H, W)
        
        lpips_vals_flat = lpips(gt_flat, rec_flat).repeat(gt_flat.shape[0]) # (B*T)
        dists_vals_flat = dists(gt_flat, rec_flat).squeeze() # (B*T)
        
        # 恢复形状 (B, T)
        lpips_vals = lpips_vals_flat.view(B, T)
        dists_vals = dists_vals_flat.view(B, T)
        
        # --- B. 时序指标 (Temporal, t vs t-1) ---
        # 我们只需要计算 t=1 到 t=T-1 (共 T-1 个间隔)
        # Current frames: [:, 1:, ...]
        # Previous frames: [:, :-1, ...]
        
        # curr_gt = gt_seq[:, 1:, :, :210, :255]   # (B, T-1, C, H, W)
        # prev_gt = gt_seq[:, :-1, :, :210, :255]  # (B, T-1, C, H, W)
        # curr_rec = rec_seq[:, 1:, :, :210, :255]
        # prev_rec = rec_seq[:, :-1, :, :210, :255]

        curr_gt = gt_seq[:, 1:, :, :, :]   # (B, T-1, C, H, W)
        prev_gt = gt_seq[:, :-1, :, :, :]  # (B, T-1, C, H, W)
        curr_rec = rec_seq[:, 1:, :, :, :]
        prev_rec = rec_seq[:, :-1, :, :, :]
        
        padder = InputPadder(curr_rec[0].shape, divisor=8) # RAFT 通常需要 8 的倍数
        curr_rec_pad = padder.pad(curr_rec[0])[0].unsqueeze(0)
        prev_rec_pad = padder.pad(prev_rec[0])[0].unsqueeze(0)
        curr_gt_pad = padder.pad(curr_gt[0])[0].unsqueeze(0)
        prev_gt_pad = padder.pad(prev_gt[0])[0].unsqueeze(0)
        # 1. 光流计算 (Input: Target=Current, Source=Prev)
        flow_rec_pad = get_flow(of_model, rearrange(curr_rec_pad, 'b t c h w -> (b t) c h w'), rearrange(prev_rec_pad, 'b t c h w -> (b t) c h w'))
        flow_gt_pad = get_flow(of_model, rearrange(curr_gt_pad, 'b t c h w -> (b t) c h w'), rearrange(prev_gt_pad, 'b t c h w -> (b t) c h w'))
        warped_rec_pad = flow_warp(rearrange(prev_rec_pad, 'b t c h w -> (b t) c h w'), flow_rec_pad) # (B, T-1, C, H, W)
        # flow shape: (B, T-1, H, W, 2)
        flow_rec = padder.unpad(flow_rec_pad)
        flow_gt = padder.unpad(flow_gt_pad)
        warped_rec = padder.unpad(warped_rec_pad)

        # 2. Warping
        # Warp prev_rec using flow_rec to align with curr_rec
        warped_rec = rearrange(warped_rec, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])
        flow_rec = rearrange(flow_rec, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])
        flow_gt = rearrange(flow_gt, '(b t) c h w -> b t c h w', t=curr_gt.shape[1])

        
        # 3. Warp PSNR
        # MSE over (C, H, W) for each sample
        mse = ((warped_rec - curr_gt) ** 2).flatten(2).mean(dim=2) # (B, T-1)
        w_psnr_vals = -10 * torch.log10(mse + 1e-8)
        
        # 4. Temporal LPIPS
        # |LPIPS(GT_t, GT_t-1) - LPIPS(Rec_t, Rec_t-1)|
        # 需要再次展平计算 LPIPS
        c_gt_flat = curr_gt.reshape(-1, C, H, W)
        p_gt_flat = prev_gt.reshape(-1, C, H, W)
        c_rec_flat = curr_rec.reshape(-1, C, H, W)
        p_rec_flat = prev_rec.reshape(-1, C, H, W)
        
        lpips_gt_t = lpips(c_gt_flat, p_gt_flat).squeeze()
        lpips_rec_t = lpips(c_rec_flat, p_rec_flat).squeeze()
        
        tlpips_vals_flat = (lpips_gt_t - lpips_rec_t).abs()
        tlpips_vals = tlpips_vals_flat.repeat(B, T-1)
        
        # 5. Flow Difference (ToF)
        # Mean abs diff over (H, W, 2)
        tof_vals = (flow_rec - flow_gt).abs().flatten(2).mean(dim=2) # (B, T-1)
        
        # --- C. 分发结果 ---
        # lpips_vals: (B, T)
        # w_psnr_vals: (B, T-1)
        
        for i in range(B):
            seq = line.replace('/', '_').replace('.', '_')
            
            # 逐帧指标 (所有 T 帧)
            lpips_dict[seq] = (lpips_vals[i].cpu().numpy().tolist())
            dists_dict[seq] = (dists_vals[i].cpu().numpy().tolist())
            
            # 时序指标 (T-1 个间隔)
            w_psnr_dict[seq] = (w_psnr_vals[i].cpu().numpy().tolist())
            tlpips_dict[seq] = (tlpips_vals[i].cpu().numpy().tolist())
            tof_dict[seq] = (tof_vals[i].cpu().numpy().tolist())

# ================= 结果汇总 =================
def safe_mean(lst):
    return np.mean(lst) if lst else 0.0

seq_list = list(lpips_dict.keys())

mean_lpips = np.round(np.mean([safe_mean(lpips_dict[k]) for k in seq_list]), 3)
mean_dists = np.round(np.mean([safe_mean(dists_dict[k]) for k in seq_list]), 3)
mean_tlpips = np.round(np.mean([safe_mean(tlpips_dict[k]) for k in seq_list]) * 1e3, 2)
mean_tof = np.round(np.mean([safe_mean(tof_dict[k]) for k in seq_list]) * 1e1, 3)
mean_warpping_psnr = np.round(np.mean([safe_mean(w_psnr_dict[k]) for k in seq_list]), 3)

# if os.path.exists(output_folder) and args.slice_id == 0:
#     print(f"Cleaning existing folder: {output_folder}")
#     shutil.rmtree(output_folder)

print("Vimeo dataset")
print("QP: ", args.qp)
print(f"Model: {model_name}, test file: {test_file}")
# print(f"psnr:{np.mean(psnr_list):.2f},psnr_avg_mse:{ -10 * math.log10(np.mean(mse_list)):.4f},ssim:{np.mean(ssim_list):.4f},psnr_LFR_lq:{np.mean(LFR_lq_psnr_list):.2f},ssim_LFR_lq:{np.mean(LFR_lq_ssim_list):.4f},psnr_LFR_hq:{np.mean(LFR_hq_psnr_list):.2f},sigma:{np.mean(sigma_list):.4f},psnr inter:{np.mean(psnr_inter_list):.2f},ssim inter:{np.mean(ssim_inter_list):.4f},lpips:{mean_lpips:.4f},dists:{mean_dists:.4f},tlpips(1e3):{mean_tlpips:.2f},tof(1e1):{mean_tof:.4f},warpping_psnr:{mean_warpping_psnr:.2f},vmaf:{np.mean(vmaf_list):.4f},ave_img_bpp:{np.mean(bpp_list):.6f}")
print(f"psnr:{np.mean(psnr_list):.2f},psnr_avg_mse:{ (np.mean(mse_list)):.10f},ssim:{np.mean(ssim_list):.4f},psnr_LFR_lq:{np.mean(LFR_lq_psnr_list):.2f},ssim_LFR_lq:{np.mean(LFR_lq_ssim_list):.4f},psnr_LFR_hq:{np.mean(LFR_hq_psnr_list):.2f},sigma:{np.mean(sigma_list):.4f},psnr inter:{np.mean(psnr_inter_list):.2f},ssim inter:{np.mean(ssim_inter_list):.4f},lpips:{mean_lpips:.4f},dists:{mean_dists:.4f},tlpips(1e3):{mean_tlpips:.2f},tof(1e1):{mean_tof:.4f},warpping_psnr:{mean_warpping_psnr:.2f},vmaf:{np.mean(vmaf_list):.4f},ave_img_bpp:{np.mean(bpp_list):.6f}")
