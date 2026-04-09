
<div align="center">

<h1>Invertible Neural Networks for Compression-aware Temporal Video Rescaling</h1>



<div>
    <a href='https://scholar.google.com/citations?user=_6xtfHYAAAAJ&hl=en&oi=ao' target='_blank'>Xinmin Feng</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/lil1/en/index.htm' target='_blank'>Li Li</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/dongeliu/en/index.htm' target='_blank'>Dong Liu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=5bInRDEAAAAJ&hl=en&oi=ao' target='_blank'>Feng Wu</a>
</div>
<div>
    Intelligent Visual Lab, University of Science and Technology of China &emsp; 
</div>

<div>
   <strong>Under review</strong>
</div>
<div>
    <h4 align="center">
    </h4>
</div>

<!-- [![icon](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2504.18398)  -->
[![python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3900/) [![pytorch](https://img.shields.io/badge/PyTorch-1.12.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/previous-versions/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=fengxinmin.TVRN_public)

---

</div>

### :hourglass: TODO
- [x] ~~Repo release~~
- [x] ~~Pretrained models of CSTVR with the proposed surrogate network~~
- [ ] Update paper link
- [x] ~~Pretrained models~~
- [ ] Code release (to be continue)


## :whale: Environment Setup

We provide a pre-configured Docker image to simplify environment setup:

```dockerfile
FROM registry.cn-hangzhou.aliyuncs.com/upr/img:add_vvc
ARG DEBIAN_FRONTEND=noninteractive
ENTRYPOINT service ssh restart && bash && source activate uprnet
```
After pulling the image, enable full `ffmpeg` and `skvideo` support by running:
```bash
cd TVRN
bash pip_opt.sh
```
### Note on VVC Support:
If you intend to use VVC, you must additionally run the following command to configure the skvideo library to locate the correct FFmpeg binary:
```bash
sed -i '23s|_FFMPEG_PATH = "/root/miniconda/envs/uprnet/bin"|_FFMPEG_PATH = "/opt/ffmpeg/bin"|' /root/miniconda/envs/uprnet/lib/python3.7/site-packages/skvideo/__init__.py
```

**Prepare the modified HEVC decoder:**
The decoder is capable of parsing motion vector fields and compression residuals from the bitstream, which are used by the surrogate network to simulate encoder distortion.
We provide two options for preparing the modified HEVC decoder:

- Compile from source code.
```bash
cd ./HEVC_decoder
# If yasm package is not installed, use the following command. 
sudo apt-get install -y yasm
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make -j9
make DESTDIR={install_path} install
```
- Use pre-compiled binary files for ubuntu 18.04 at `utils/hevc.bin`. 

## :open_book: Pretrained Model Weight
### a. Retrained CSTVR Models
To validate the effectiveness of our approach, we retrained CSTVR using the proposed surrogate network, following the original training setup.

You can download the retrained models from our release page: [CSTVR Retrained Models](https://github.com/fengxinmin/TVRN_public/releases/tag/CSTVR)

### b. Pretrained TVRN Models
We provide the pretrained TVRN models on our release page:  [TVRN Pretrained Models](https://github.com/fengxinmin/TVRN_public/releases/tag/TVRN-full)



## Quick Start
We have integrated code from EBME, EMA, GIMM, RIFE, CVRS, and other works in this repository to facilitate benchmark testing. You only need to run:
```bash 
bash test_vimeo.sh
bash test_ucf.sh
bash test_snu.sh
```
You can adjust the number of frames and other parameters via the bash scripts according to your hardware capabilities. The format is as follows:
```bash
bash run_benchmark.sh [DATASET_NAME] [CODEC_NAME] [METHOD_NAME]  [QPS_LIST] [GPU_NUM] [OPTIONAl:65frames]
# DATASET_NAME: SNU, UCF101, vimeo90k
# CODEC_NAME: hevc, av1, vvc, vp9, avc
# METHOD_NAME: TVRN, EMA, GIMM, STAA, IFRNet, RIFE, codec_reference
# QPS_LIST: 17,22,27,...
```
For example, to test TVRN on 65-frame SNU_FILM sequences using VVC across 8 GPUs:
```bash 
bash run_benchmark.sh SNU vvc TVRN  19,23,28,33  8  65frames
```



