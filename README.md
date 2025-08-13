<div align="center">
<h1>AU-Net: Adaptive Unified Network for </br> Joint Multi-modal Image Registration and Fusion</h1>

[**Ming Lu**](https://luming1314.github.io/),  Min Jiang, Xuefeng Tao, Jun Kong <br>

Jiangnan University

<!-- <sup>*</sup>corresponding authors -->

<a href='https://doi.org/10.1109/TIP.2025.3586507'><img src='https://img.shields.io/badge/DOI-10.1109%2FTIP.2025.3586507-blue'></a>

</div>

This repository provides the **Image-to-Image Translation module** of AU-Net. For the registration and fusion code, please refer to the main [AU-Net](https://github.com/luming1314/AU-Net) repository.

## ✨ Usage

### Quick start
#### 1. Clone this repo and setting up environment
```sh
git clone https://github.com/luming1314/I2I-DDPM.git
cd I2I-DDPM
conda create -n I2I-DDPM python=3.8 -y
conda activate I2I-DDPM
pip install -r requirements.txt
```

#### 2. Download pre-trained models

You can download our pre-trained models for a quick start.

Baidu Netdisk | Description
| :--- | :----------
[I2I-DDPM](https://pan.baidu.com/s/1f9lOUNzmC5ybfN-6YUox5A?pwd=gias) |Pre-trained I2I-DDPM model

#### 3. Test

To test I2I-DDPM, execute `run_test.sh`:

```shell
sh run_test.sh
```

## ⚙️ Training

### Prepare data
I2I-DDPM is trained on the [NirScene](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) dataset. We recommend using our preprocessed dataset for training:

| Baidu Netdisk| Description
| :--- |:----------
|[Image-to-image translation](https://pan.baidu.com/s/1KhdKYnwleQIENTVOLrd1gw?pwd=4i8a) | Training and Testing Datasets for I2I-DDPM

To train your own model, adapt your dataset to match the structure shown below:

### Dataset structure
```markdown
📦 datasets
├── 📂 test                # Test Dataset
│   ├── 📂 TH              
│   └── 📂 VIS             
├── 📂 val                 # Val Dataset
│   ├── 📂 TH              
│   └── 📂 VIS             
└── 📂 train               # train Dataset
    ├── 📂 TH              # Infrared images
    └── 📂 VIS             # Visible images                               
```
### Training I2I-DDPM
To train I2I-DDPM, execute `run_train.sh`:
```shell
sh run_train.sh
```

## 👏 Acknowledgment
Our work is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:
* SuperFusion: https://github.com/Linfeng-Tang/SuperFusion
* T2V-DDPM: https://github.com/Nithin-GK/T2V-DDPM
* ODConv: https://github.com/OSVAI/ODConv
## 🎓 Citation

If AU-Net is helpful to your work, please cite our paper via:

```
@ARTICLE{11079838,
  author={Lu, Ming and Jiang, Min and Tao, Xuefeng and Kong, Jun},
  journal={IEEE Transactions on Image Processing}, 
  title={AU-Net: Adaptive Unified Network for Joint Multi-Modal Image Registration and Fusion}, 
  year={2025},
  volume={34},
  number={},
  pages={4721-4735},
  doi={10.1109/TIP.2025.3586507}
  }

```