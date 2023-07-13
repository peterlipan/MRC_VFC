#!/bin/bash

# NV - nevus 12875
isic image download --search 'diagnosis:"nevus"' --limit 12875 /mnt/ssd/li/ISIC_Archive
# rename the metadata
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/NV.csv

# MEL - melanoma 4522
isic image download --search 'diagnosis:"melanoma"' --limit 4522 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/MEL.csv

# BCC - basal cell carcinoma 3393
isic image download --search 'diagnosis:"basal cell carcinoma"' --limit 3393 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/BCC.csv

# SK - seborrheic keratosis 1464
isic image download --search 'diagnosis:"seborrheic keratosis"' --limit 1464 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/SK.csv

# AK - actinic keratosis 869
isic image download --search 'diagnosis:"actinic keratosis"' --limit 869 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/AK.csv

# SCC - squamous cell carcinoma 656
isic image download --search 'diagnosis:"squamous cell carcinoma"' --limit 656 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/SCC.csv

# BKL - pigmented benign keratosis 384
isic image download --search 'diagnosis:"pigmented benign keratosis"' --limit 384 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/BKL.csv

# SL - solar lentigo 270
isic image download --search 'diagnosis:"solar lentigo"' --limit 270 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/SL.csv

# VASC - vascular lesion 253
isic image download --search 'diagnosis:"vascular lesion"' --limit 253 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/VASC.csv

# DF - dermatofibroma 246
isic image download --search 'diagnosis:"dermatofibroma"' --limit 246 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/DF.csv

# LK - lichenoid keratosis 32
isic image download --search 'diagnosis:"lichenoid keratosis"' --limit 32 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/LK.csv

# LS - lentigo simplex 27
isic image download --search 'diagnosis:"lentigo simplex"' --limit 27 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/LS.csv

# AN - angioma 15
isic image download --search 'diagnosis:"angioma"' --limit 15 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/AN.csv

# AMP - atypical melanocytic proliferation 14
isic image download --search 'diagnosis:"atypical melanocytic proliferation"' --limit 14 /mnt/ssd/li/ISIC_Archive
mv /mnt/ssd/li/ISIC_Archive/metadata.csv /mnt/ssd/li/ISIC_Archive/AMP.csv

python3 merge.py
