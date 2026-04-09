# ===================
# 制作用于compression-aware模块的训练数据集
# ===================

import os
import math
import argparse
import random
import logging
import numpy as np
from einops import rearrange
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
import lmdb
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
# from models import create_model
# from data.util import rgb2ycbcr_tensor
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
from  data.util import get_image_paths, read_img
import matplotlib.pyplot as plt     


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


# from tqdm import tqdm

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--qp', type=int, default=22, help='QP value')
    args = parser.parse_args()
    
    # 解析配置文件
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    
    print("target file: ", f"/data/fengxm/vimeo90k/vimeo90k_train_Q{args.qp}.lmdb")
    
    # 打开LMDB环境
    map_size = 100 * 1024 * 1024 * 1024  # 10GB in bytes
    source_env = lmdb.open(f"/data/fengxm/vimeo90k/vimeo90k_train_GT.lmdb", readonly=False, lock=False, readahead=False, meminit=False, map_size=map_size)
    target_env = lmdb.open(f"/data/fengxm/vimeo90k/vimeo90k_train_Q{args.qp}.lmdb", readonly=False, lock=False, readahead=False, meminit=False, map_size=map_size)

    # 获取图像路径
    paths_GT, _ = get_image_paths('lmdb', f"/data/fengxm/vimeo90k/vimeo90k_train_GT.lmdb")
    data_len = len(paths_GT)
    
    # 初始化H.265编解码器
    opt["network_G"]["h265_all_default"] = False
    Quantization_H265_codec = Quantization_H265_Stream(args.qp, -1, None, opt)
    
    # 处理每张图片
    for index in (range(data_len)):
        print(index)
        key = paths_GT[index]
        name_a, name_b = key.split('_')
        
        img_GT_l = []
        for v in range(1,8):
            img_GT = read_img(source_env, name_a + '_{}'.format(v), (3, 256, 448))
            img_GT_l.append(img_GT)
        
        img_GTs = np.stack(img_GT_l, axis=0)
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]  # BGR to RGB
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        
        # 对图像进行处理（这里是示例，你可以根据需求进行更复杂的处理）
        output_array = np.zeros_like(img_GTs)
        for b in range(img_GTs.shape[0]):
                rgb_image = rearrange(img_GTs[b], 'c h w -> h w c')
                yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
                output_array[b] = rearrange(yuv_image, 'h w c -> c h w')
        
        real_H = torch.tensor(output_array)
        
        # 示例：使用随机数据代替实际的处理过程
        t, c, h, w = real_H.shape
        ppppp = torch.randn(t, 1, w, h)
        
        # 打开H.265编码器的写入器
        Quantization_H265_codec.open_writer(ppppp.cpu().device, w, h, pix_fmt='yuv444p', verbosity=0)
        
        # 写入处理后的数据到编码器
        for i in range(t):
            output = rearrange(real_H[i:i+1], 't c h w -> (t c) h w').unsqueeze(0)
            Quantization_H265_codec.write_multi_frames(output)
        
        # 关闭编码器的写入器，并获取编码后的数据
        _, img_distri = Quantization_H265_codec.close_writer()
        
        # 示例：打开编码器的读取器，并读取编码后的数据（这部分根据实际需求进行修改）
        Quantization_H265_codec.open_reader(verbosity=0)
        outsouts2 = []
        for i in range(t):
            v_seg = Quantization_H265_codec.read_multi_frames(1)
            outsouts2 += [v_seg]
        outout2 = torch.cat(outsouts2, dim=0)
        
        # 将处理后的数据写回LMDB文件
        with target_env.begin(write=True) as txn:
            for v in range(1,8):
                img = (outout2[v - 1].numpy() * 255.).astype(np.uint8)
                img = rearrange(img, 'c h w -> h w c')
                img_bytes = img.tobytes()
                txn.put(f'{name_a}_{v}'.encode('ascii'), img_bytes)
        
    
if __name__ == '__main__':
    main()
