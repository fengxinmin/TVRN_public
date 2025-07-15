
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
- [ ] Pretrained models
- [ ] Code release


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
**Prepare HEVC feature decoder:**
The decoder is capable of parsing motion vector fields and compression residuals from the bitstream, which are used by the surrogate network to simulate encoder distortion.
We provide two options for preparing the modified HEVC decoder:

- Compile from source code (openHEVC_feature_decoder).
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

## :open_book: Retrained CSTVR Models
To validate the effectiveness of our approach, we retrained CSTVR using the proposed surrogate network, following the original training setup.

You can download the retrained models from our release page:
👉 [CSTVR Retrained Models](https://github.com/fengxinmin/TVRN_public/releases/tag/CSTVR)
