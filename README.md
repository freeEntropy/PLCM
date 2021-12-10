# Pseudo-loss Confidence Metric for Semi-supervised Few-shot Learning
This is the implementation of the paper "Pseudo-loss Confidence Metric for Semi-supervised Few-shot Learning" (accepted to ICCV2021).
## Enviroment
### Software
- Ubuntu 18.04
- python3
- pytorch 1.3
- sklearn
### Hardware
- Intel Xeon E5-2697 v3
- RTX 2080TI

## Prepare Datasets
Please make sure the datasets follows the structure as following ([process tools](https://github.com/yaoyao-liu/mini-imagenet-tools)), all the processed data can be downloaded from our Baidu Cloud: [[link](https://pan.baidu.com/s/1plz04xZ10sVThYoAO4TryA)][access code 4ud8] or Google Cloud: [[link](https://drive.google.com/drive/folders/1RXrl44bpXSA3LMl7bBke_pQI-Xqf8i9y?usp=sharing)]
```
|--dataset name
    |-- images (folder, contains all images)
    |-- train.csv (csv file, records the names and labels of training data)
    |-- val.csv (csv file, records the names and labels of validation data)
    |-- test.csv (csv file, records the names and labels of evaluation data)
```
For convenience, you could link the datasets to the source code folder:
```
ln -s path_to_datasets data
```

## Train
- step 1: pre-train the embedding network.

```
python train_embedding.py --dataset dataset_name
```
- step 2: process PLCM (5-way 1-shot 30-unlabel as example).

```
python main.py --dataset dataset_name --way 5 --shot 1 --unlabel 30
```

## Usage for Pre-trained Models
We also provided the pre-trained models to skip the training of step 1, which are stored in ckpt folder.
Baidu Cloud: [[link](https://pan.baidu.com/s/1PBMvL_xYiCx0ycZpp7Q-4Q)][access code ujux] or Google Cloud: [[link](https://drive.google.com/drive/folders/1d7lwJtdqi8qRD6Tx_zcaTI7roI4We6WL?usp=sharing)]

## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@inproceedings{huang2021pseudo,
  title={Pseudo-Loss Confidence Metric for Semi-Supervised Few-Shot Learning},
  author={Huang, Kai and Geng, Jie and Jiang, Wen and Deng, Xinyang and Xu, Zhe},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8671--8680},
  year={2021}
}
````
