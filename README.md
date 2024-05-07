# Mult-scale-SAM
## Introduction

**MSA-SAM** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on developing advanced Vision Transformers for remote sensing image segmentation.

## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.
  
- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.
  
- Supported Remote Sensing Datasets
 
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [UAVid](https://uavid.nl/)
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - More datasets will be supported in the future.
  
- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images

## Supported Networks

- Vision Transformer

  - [UNetFormer](https://authors.elsevier.com/a/1fIji3I9x1j9Fs) 
  - [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)
  - [CMTFNet](https://ieeexplore.ieee.org/document/10247595)
  - [RS3Mamba](https://arxiv.org/abs/2404.02457)
  - [RSAM-Seg](https://arxiv.org/abs/2402.19004)
  
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── MSA-SAM (code)
├── pretrained_models (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install


## Data Preprocessing

Download the datasets from the official website and split them yourself. We adopt the data processing method in the UNetFormer(https://authors.elsevier.com/a/1fIji3I9x1j9Fs) 

## Training
```
python MSA-SAM/train.py --AdamW
```
## Testing

```
python MSA-SAM/vaihingen_test.py --rgb -t 'd4'
```

## Reproduction Results
|    Method     |  Dataset  |  F1   |  OA   |  mIoU |
|:-------------:|:---------:|:-----:|:-----:|------:|
|    MSA-SAM    |   UAVid   |   -   | 89.7  | 76.3  |
|    MSA-SAM    | Vaihingen | 92.3  | 94.1  | 86.0  |
|    MSA-SAM    |  Potsdam  | 93.6  | 92.5  | 88.2  |
|    MSA-SAM    |  LoveDA   |   -   |   -   | 54.8  |

<div>
<img src="figs/vaihingen_visualization.png" width="30%"/>
<img src="figs/potsdam_visualization.png" width="35.5%"/>
<img src="figs/uavid_visualization.png" width="35.5%"/>
<img src="figs/loveda_visualization.png" width="35.5%"/>
</div>
