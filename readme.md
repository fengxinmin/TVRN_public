
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
<!-- [![python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3900/) [![pytorch](https://img.shields.io/badge/PyTorch-1.12.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/previous-versions/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ustcivclab.IPM) -->

---

</div>

### :hourglass: TODO
- [x] ~~Repo release~~
- [x] ~~Pretrained models of CSTVR with the proposed surrogate network~~
- [ ] Update paper link
- [ ] Pretrained models
- [ ] Code release


## :open_book: Retrained CSTVR models

To validate the effectiveness of our approach, we retrain CSTVR using our proposed surrogate network, following [the original training settings](https://github.com/hahazh/CSTVR).
The retrained models are available at our release: https://github.com/fengxinmin/TVRN_public/releases/tag/CSTVR






<!-- - [Installation of Dependencies](#installation-of-dependencies)
- [Quick Start](#raised_hand-quick-start)
- [Training Dataset](#training-dataset)
- [Modified VTM Encoder](#modified-vtm-encoder) -->


<!-- ⭐ If this project is helpful to you, please star this repo. Thanks! 🤗 -->
<!-- 
## :wrench: Installation of Dependencies

In order to explore this project, it is needed to first install the libraries used in it.

The base image is `pytorch:2.0.0-cuda11.7-cudnn8-runtime`. To install the dependencies, use the following command:

```bash
pip install einops matplotlib tensorboard timm ipykernel h5py thop openpyxl palettable -i https://mirrors.aliyun.com/pypi/simple/
```


## :raised_hand: Quick Start

The evaluation involves two steps:

### Step 1: Network Inference
Run the neural network on raw YUV sequences to generate partition flags. The post-processing algorithm then converts these into a format compatible with the modified VTM encoder.

**Preparation**:
Place the test sequences (e.g., `BasketballDrive_1920x1080_50.yuv`) in the `./test_sequences/` directory.

Run Inference:
```bash
python model_inference.py -opt ./options/test/test_baseline.yml -qp 32
```
The output partition matrices will be saved to `./results/PartitionMat/`.

### Step 2: Run the Modified VTM Encoder

Employ the modified VTM encoder to compress the sequences using the predicted partition flags.

> **Note:**
> We provide a precompiled Windows binary in `./codec/exe/`.  
> For Linux users, please compile the encoder from source located in `./codec/exe/source_code/`.

**Command:**
```bash
[EXE_NAME].exe -el [PARTITION_MAT_PATH] -c [RANDOM_ACCESS_CFG_PATH] -c [SEQUENCE_CFG_PATH] -i [YUV_SEQUENCE_PATH] -q [QP] -f [FRAME_NUM] -ip [INTRA_PERIOD] -b [OUTPUT_BIN_PATH]
```

Example
```
VTM10_L1_20_90.exe -el D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\encoder_randomaccess_vtm.cfg -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\per-sequence\\BasketballDrive.cfg -i D:\\VVC_test\\BasketballDrive_1920x1080_50.yuv  -q 22 -f 65 -ip 48 -b res_L0.bin
```



##  :open_book: Training Dataset

The training dataset is available at [Baidu Cloud](https://pan.baidu.com/s/1ZMPZqOcQS_gri_pzSq2vGA?pwd=tmxn). We used 668 4K sequences with 32 frames from the [BVI-DVC](https://fan-aaron-zhang.github.io/BVI-DVC/) dataset, Tencent Video Dataset / [TVD](https://multimedia.tencent.com/resources/tvd/), and [UVG](https://github.com/ultravideo/UVG-4K-Dataset/) dataset. These sequences were cropped or downsampled to create datasets with four different resolutions: 3840x2160, 1920x1080, 960x544, and 480x272. We organized the training dataset using HDF5 format, which includes the following files:

- `train_seqs.h5`: Luma components of the original sequences.
- `train_qp22.h5`: Training dataset label for basic QP22.
- `train_qp27.h5`: Training dataset label for basic QP27.
- `train_qp32.h5`: Training dataset label for basic QP32.
- `train_qp37.h5`: Training dataset label for basic QP37.

To further support subsequent research, we also provide the code for generating the training dataset, which includes:

1. Modified VTM source code `codec/print_encoder` and the executable file `codec/exe/print_encoder.exe` for extracting block partitioning statistics from YUV sequences. Code `dataset_preparation.py` for extracting the statistics into `DepthSaving/` with multiple threads.
3. Code `depth2dataset.py` for converting the  statistics into partition maps.


## :gear: Modified VTM Encoder

We provide the source code for the VTM 10.0 and 23.0 encoder with integrated fast algorithms in the folder `codec/source_code/inter_fast`, and the corresponding executable files for different acceleration levels in `codec/exe`. Specifically, `inter_fast` corresponds to acceleration for B-frames only, while `inter_intra_fast` uses the proposed method to accelerate B-frames and uses the method from [1] to accelerate I-frames.

To implement different acceleration levels, you can modify the parameters in `TypeDef.h`. For example, for the acceleration level $L_1(0.2,0.9)$, and the configuration for accelerating I-frames is as follows:

```C++
// Fast block partitioning for VVC inter coding
#define   INTER_PARTITION_MAP_ACCELERATION_FXM      1  // Accelerating B-frames, True: 1, False: 0
#define   Acceleration_Config_fxm                   1  // Acceleration level, options: 0, 1, 2, 3
#define   boundary_handling_fxm                     1  // Boundary handling based on granularity
#define   Mtt_mask_fxm                              1  // If config=0 and mtt_mask=1, the uncovered parts of the mtt mask are decided by RDO. If config>=1 and mtt_mask=1, the uncovered parts are decided by the network
#define   mtt_mask_thd                              20 // MTT mask threshold, true threshold = threshold / 100
#define   mtt_rdo_thd                               90 // MTT RDO threshold. Blocks with values below this will skip MTT fast partitioning

// Fast block partitioning for VVC intra coding
#define   INTRA_PARTITION_MAP_ACCELERATION_FAL      1  // Accelerating I-frames, True: 1, False: 0
#if INTRA_PARTITION_MAP_ACCELERATION_FAL
#define   Acceleration_Config_fal_intra             1  // 4 configuration options (0, 1, 2, 3)
#endif
```

The acceleration configurations for different acceleration levels are as follows, , corresponding to `inter_fast/VTM10_L0_0_100.exe`, `inter_fast/VTM10_L0_20_100.exe`, and `inter_fast/VTM10_L1_20_90.exe`.

| Macro                            | $L_0(0,1)$ | $L_0(0.2,1)$ | $L_0(0.2,0.9)$ |
|-----------------------------------|------------|--------------|----------------|
| `INTER_PARTITION_MAP_ACCELERATION_FXM` | 1          | 1            | 1              |
| `Acceleration_Config_fxm`           | 0          | 0            | 1              |
| `boundary_handling_fxm`             | 1          | 1            | 1              |
| `Mtt_mask_fxm`                      | 0          | 1            | 1              |
| `mtt_mask_thd`                      | 0          | 20           | 20             |
| `mtt_rdo_thd`                       | 100        | 100          | 90             |

In addition, we also provide a combination of the proposed method and previous work [1], where the former accelerates B-frames and the latter accelerates I-frames. This corresponds to `inter_intra_fast/VTM10_L0i_0_100.exe`, `inter_intra_fast/VTM10_L0i_20_100.exe`, and `inter_intra_fast/VTM10_L1i_20_90.exe`.

You can use the following command to run the encoder and accelerate B-frames, where `el` represents the path to the partition flags of B-frames, and `ip` represents the intra period.

```bash
VTM10_L1_20_90.exe -el D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\encoder_randomaccess_vtm.cfg -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\per-sequence\\BasketballDrive.cfg -i D:\\VVC_test\\BasketballDrive_1920x1080_50.yuv  -q 22 -f 65 -ip 48 -b res_L0.bin
```
Alternatively, you can use the following command to accelerate both B-frames and I-frames. In this case, `ac` and `al` represent the paths to the partition flags for the I-frame Luma components and chroma components, respectively.

```bash
VTM10_L1_20_90.exe -el D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -ac D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -al D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_intra\\RitualDance_1920x1080_60fps_10bit_420_Luma_QP22_PartitionMat_intra.txt  -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\VVCSoftware_VTM-VTM-10.0-fast\\cfg\\encoder_randomaccess_vtm.cfg -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\VVCSoftware_VTM-VTM-10.0-fast\\cfg\\per-sequence\\RitualDance.cfg -i E:\\VVC_test\\RitualDance_1920x1080_60fps_10bit_420.yuv  -q 22 -f 65 -ip 64 -b res_L0.bin
```

We provide partition flags for 22 VVC CTC sequences in GOP16 and GOP32 on [Baidu Cloud](https://pan.baidu.com/s/1STnEpLJmxiVV8AoA3hptRA?pwd=dddr). You can download these files and replace the `el`, `ac`, and `al` paths above to reproduce our results without invoking model.





## :bust_in_silhouette: Citing

If you use our work in your research, please use the following BibTeX entry.
```
@article{feng2025partition,
  title={Partition Map-Based Fast Block Partitioning for VVC Inter Coding},
  author={Feng, Xinmin and Li, Zhuoyuan and Li, Li and Liu, Dong and Wu, Feng},
  journal={arXiv preprint arXiv:2504.18398},
  year={2025}
}
```

```
@article{feng2023partition,
  title={Partition map prediction for fast block partitioning in VVC intra-frame coding},
  author={Feng, Aolin and Liu, Kang and Liu, Dong and Li, Li and Wu, Feng},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={2237--2251},
  year={2023},
  publisher={IEEE}
}
``` -->