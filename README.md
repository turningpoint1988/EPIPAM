# EPIPAM
Implementation of "Predicting enhancer-promoter interactions using deep neural network based on position attention mechanism"

## Requirements

+ Pytorch 1.1 
+ Python 3.6
+ CUDA 9.0
+ Keras 2.2
+ Tensorflow 1.5
+ Python packages: biopython, sklearn

## Data preparation
(1) Downloading hg19.fa from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/, and put it into /hg19.

(2) Encoding EPIs datasets by using embeding or one-hot.
+ Usage:
  ```
  cd /EPIs
  ln -s /yourpath/EPIPAM/hg19 hg19
  bash Prepare.sh <command>
  ```
  **'command'** has two options, including 'embed' and 'seq'.
  
 (3) Encoding ChIA-PET datasets by using embeding or one-hot.
+ Usage:
  ```
  cd /ChIA-PET
  ln -s /yourpath/EPIPAM/hg19 hg19
  bash Prepare.sh <command>
  ```
  **'command'** has two options, including 'embed' and 'seq'.
  
 (4) Building soft links to EPIs and ChIA-PET datasets.
+ Usage:
  ```
  ln -s /yourpath/EPIPAM/EPIs /EPIPAM_onehot/EPIs
  ln -s /yourpath/EPIPAM/ChIA-PET /EPIPAM_onehot/ChIA-PET
  ln -s /yourpath/EPIPAM/EPIs /EPIVAN/EPIs
  ln -s /yourpath/EPIPAM/ChIA-PET /EPIVAN/ChIA-PET
  ln -s /yourpath/EPIPAM/EPIs /DeepTACT/EPIs
  ln -s /yourpath/EPIPAM/ChIA-PET /DeepTACT/ChIA-PET
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

