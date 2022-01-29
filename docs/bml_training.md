# nnUnet training task04_Hippocampus in BML

## 一、install pytorch
pytorch >= 1.6

## 二、install nnUNet
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
```

## 三、set path
```bash
export nnUNet_raw_data_base="/media/fabian/nnUNet_raw_data_base"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"
```

## 四、preprocess
```bash
nnUNet_plan_and_preprocess -t 4
```

## 五、training
BML直接训练可能会报错“RuntimeError: unable to write to file </torch_833_3533296244>”，可能是由于多线程引起的
修改nnunet/training/data_augmentation/data_augmentation_moreDA.py

1、导入SingleThreadedAugmenter    
``
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
``

2、将用到该文件内SingleThreadedAugmenter的2行代码取消注释  

训练：  
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 4 0
```

