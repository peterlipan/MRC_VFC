# MRC-VFC

This is the official repo of our paper *Combat Long-tails in Medical Classification with Relation-aware Consistency and Virtual Features Compensation*.



## 1. Dataset preparation

To start with, download the official ISIC datasets and split them into train/val/test:

```bash
# ISIC 2019
bash ./prepare_datasets/ISIC2019LT/download_ISIC2019.sh

# ISIC Archive
bash ./prepare_datasets/ISIC_Archive/download_isic_archive.sh
python merge.py
```



## 2. Stage 1 

Execute the first stage of training of our framework. 

```bash
python stage1.py
```



## 3. Stage 2

Second stage training and evaluation.

```bash
python stage2.py
```

