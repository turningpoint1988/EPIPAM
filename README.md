# EPIPAM
Implementation of "Predicting enhancer-promoter interactions using deep neural network based on position attention mechanism"

## Requirements

+ Pytorch 1.1 
+ Python 3.6
+ CUDA 9.0

## Data preparation
(1) Downloading hg19.fa from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/, and put it into /hg19.

(2) Encoding EPIs datasets by using embeding or one-hot.
+ Usage:
  ```
  bash /EPIs/Prepare.sh <command>
  ```
  **'command'** has two options, including 'embed' and 'seq'.
  
 (3) Encoding ChIA-PET datasets by using embeding or one-hot.
+ Usage:
  ```
  bash /ChIA-PET/Prepare.sh <command>
  ```
  **'command'** has two options, including 'embed' and 'seq'.
  
  (4) Building soft links to EPIs and ChIA-PET datasets.
+ Usage:
  ```
  ln -s /EPIs /EPIPAM_embed/EPIs
  ln -s /ChIA-PET /EPIPAM_embed/ChIA-PET
  ln -s /EPIs /EPIPAM_onehot/EPIs
  ln -s /ChIA-PET /EPIPAM_onehot/ChIA-PET
  
  ```

## Run 
**Run EPIPAM with embeding encoding**
+ Usage: 
  ```
  cd /EPIPAM_embed
  bash train.sh <dataset>
  ```
 **'dataset'** has two options, including 'EPIs' and 'ChIA-PET'.
 
**Run EPIPAM with one-hot encoding**
+ Usage: 
  ```
  cd /EPIPAM_onehot
  bash train.sh <dataset>
  ```
 **'dataset'** has two options, including 'EPIs' and 'ChIA-PET'.

