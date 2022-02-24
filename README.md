# nnUnet_exp

[1、BML训练nnUNet](docs/bml_training.md)


# 一、目录介绍
|-nnUNet_paddle   包含生成plans以及其他配置文件的所有代码     
|-docs            文档  
|-test.py		  生成plans以及其他配置文件的示例代码，以task004数据集为例，需要更改文件路径  

# 二、使用方法
1、数据集转换  
```
python nnUNet_paddle/experiment_planning/convert_decathlon_task.py -i ${Path_to_raw_data_dir} -o {splited_data_save_dir}
```
示例： 
```
python nnUNet_paddle/experiment_planning/convert_decathlon_task.py -i ~/data/data126985/Task04_Hippocampus -o ~/converted_data/
```

2、裁剪数据集、生成配置等     
test.py以task004数据集为例，生成相关配置文件，在运行前，先在代码中修改数据集路径。
```
python test.py
```


# aistudio  
AISTUDIO：https://aistudio.baidu.com/aistudio/projectdetail/3451532?contributionType=1