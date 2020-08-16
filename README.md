# EPIPAM
Implementation of "Predicting enhancer-promoter interactions using deep neural network based on position attention mechanism"

## Requirements

+ Pytorch 1.1 
+ Python 3.6
+ CUDA 9.0

## Data preparation
Firstly, downloading hg19.fa from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/
Secondly, encoding EPIs datasets by using embeding or one-hot.
+ Usage:
  ```
  bash /EPIs/Prepare.sh <command>
  ```
  **<command>** has two options, including 'embed' and 'seq'.

## Run 
**Run DeepBind_K or DeepCNN without using DNA shape information**
+ Usage: you can excute run.sh script directly, in which you should modify the python command accordingly, e.g.:
  ```
  python train_val_test.py -datadir ./pbmdata/$eachTF/data -run 'noshape' -model 'shallow' -batchsize 300 -k 5 -params 30 --train
  ```
 The command '-model' can be a choice of {'shollow', 'deep'}, where 'shollow' means DeepBind_K, and 'deep' means DeepCNN.
 
**Run DLBSS(shallow) or DLBSS(deep) using DNA shape information**
+ Usage: you can excute run.sh script directly, in which you should modify the python command accordingly, e.g.:
  ```
  python train_val_test_hybrid.py -datadir ./pbmdata/$eachTF/data -run 'shape' -model 'shallow' -batchsize 300 -k 5 -params 30 --train
  ```
The command '-run' can be a choice of {'shape', 'MGW', 'ProT', 'Roll', 'HelT'}, where 'shape' means using all shape features, 'MGW' means using MGW shape feature, and so on.<br />
The command '-model' can be a choice of {'shollow', 'deep'}, where 'shollow' means DLBSS(shallow), and 'deep' means 'DLBSS(deep)'.<br />
**Note that** you should change the ouput path in the run.sh script, the naming rule is: 'model_' + args.model + '_' + args.run.

+ Type the following for details on other optional arguments:
	```
  python train_val_test_hybrid.py -h
