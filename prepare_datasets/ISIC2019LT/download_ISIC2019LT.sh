#!/bin/bash
wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

mkdir /mnt/ssd/li/ISIC2019LT

unzip -jn ISIC_2019_Training_Input.zip -d /mnt/ssd/li/ISIC2019LT
mv ISIC_2019_Training_GroundTruth.csv /mnt/ssd/li/ISIC2019LT

rm ISIC_2019*.zip
