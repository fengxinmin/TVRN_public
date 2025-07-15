#!/bin/bash

### Create required directories ###
mkdir -p /root/.cache/torch/hub/checkpoints/
mkdir -p /output/recon/
mkdir -p /output/orig/

### Configure conda mirrors ###
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

### Fix FFMPEG path for sk-video ###
# (Optional) Ensure sk-video is installed first
# conda install sk-video
sed -i '23s|_FFMPEG_PATH = "/root/ffmpeg"|_FFMPEG_PATH = "/root/miniconda/envs/uprnet/bin"|' \
    /root/miniconda/envs/uprnet/lib/python3.7/site-packages/skvideo/__init__.py

### Replace alexnet with custom version and copy pretrained models ###
cp /code/codes/utils/alexnet.py \
    /root/miniconda/envs/uprnet/lib/python3.7/site-packages/torchvision/models/alexnet.py

cp /data/fengxm/vimeo90k/pretrained_model/alexnet-owt-4df8aa71.pth \
    /root/.cache/torch/hub/checkpoints/

cp /data/fengxm/vimeo90k/pretrained_model/weights-inception-2015-12-05-6726825d.pth \
    /root/.cache/torch/hub/checkpoints/

cp /data/fengxm/vimeo90k/pretrained_model/pt_inception-2015-12-05-6726825d.pth \
    /root/.cache/torch/hub/checkpoints/

cp /data/fengxm/vimeo90k/pretrained_model/vgg16-397923af.pth \
    /root/.cache/torch/hub/checkpoints/

### Replace FFMPEG Python bindings to support YUV420P ###
cp /code/codes/utils/ffmpeg.py \
    /root/miniconda/envs/uprnet/lib/python3.7/site-packages/skvideo/io/ffmpeg.py

### Install essential Python packages ###
pip install opencv-python-headless tqdm \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install /model/fengxm/VRN/python_extension/mmcv-2.0.0rc3-cp37-cp37m-manylinux1_x86_64.whl \
    -i https://mirrors.aliyun.com/pypi/simple

pip install omegaconf yacs easydict \
    -i https://mirrors.aliyun.com/pypi/simple

### Optional metric tools ###
# pip install torch_fidelity dists-pytorch lpips \
#     -i https://pypi.tuna.tsinghua.edu.cn/simple

### Copy VMAF model files ###
cp /data/fengxm/vimeo90k/pretrained_model/ffmpeg_tools/vmaf-2.3.1.tar.gz /tmp
cd /tmp && tar -xzvf vmaf-2.3.1.tar.gz
cp -r vmaf-2.3.1/model /usr/local/share/model

### Extract ffmpeg static build ###
cp /data/fengxm/vimeo90k/pretrained_model/ffmpeg_tools/ffmpeg-git-amd64-static.tar.xz /tmp
cd /tmp && tar -xJvf ffmpeg-git-amd64-static.tar.xz
# ffmpeg binary located at: /tmp/ffmpeg-git-*/ffmpeg

### Ensure binary is executable ###
chmod 777 /code/codes/utils/hevc.bin

### Set environment variables ###
export LD_PRELOAD=/root/miniconda/envs/uprnet/lib/libstdc++.so.6.0.32

# Uncomment below if using momo or related Rust-based tools
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source $HOME/.cargo/env
# pip install omegaconf packaging transformers torchmetrics pyiqa accelerate einops huggingface_hub bitsandbytes diffusers \
#     -i https://mirrors.aliyun.com/pypi/simple
