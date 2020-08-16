import os
import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest']


class EPIDataSetTrain(data.Dataset):
    def __init__(self, root, name, cv):
        super(EPIDataSetTrain, self).__init__()
        self.root = root
        self.name = name
        train = np.load(osp.join(self.root, '{}_train_fold{}.npz'.format(self.name, cv)))
        self.enhancer, self.promoter, self.label = train['X_en_tr'], train['X_pr_tr'], train['y_tr']

        assert len(self.enhancer) == len(self.label) and len(self.promoter) == len(self.label), \
            "the number of sequences and labels must be consistent."

        total_num = len(self.label)
        pos_num = sum(self.label.reshape(-1) == 1)
        neg_num = sum(self.label.reshape(-1) == 0)
        print("Total train number: {} =  Positive number: {} + Negative number: {}.".format(total_num, pos_num, neg_num))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        enhancer_one = self.enhancer[index]
        promoter_one = self.promoter[index]
        label_one = self.label[index]

        return {"enhancer": enhancer_one, "promoter": promoter_one, "label": label_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, root, name, cv):
        super(EPIDataSetTest, self).__init__()
        self.root = root
        self.name = name
        test = np.load(osp.join(self.root, '{}_test_fold{}.npz'.format(self.name, cv)))
        self.enhancer, self.promoter, self.label = test['X_en_te'], test['X_pr_te'], test['y_te']

        assert len(self.enhancer) == len(self.label) and len(self.promoter) == len(self.label), \
            "the number of sequences and labels must be consistent."
        total_num = len(self.label)
        pos_num = sum(self.label.reshape(-1) == 1)
        neg_num = sum(self.label.reshape(-1) == 0)
        print("Total test number: {} =  Positive number: {} + Negative number: {}.".format(total_num, pos_num, neg_num))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        enhancer_one = self.enhancer[index]
        promoter_one = self.promoter[index]
        label_one = self.label[index]

        return {"enhancer": enhancer_one, "promoter": promoter_one, "label": label_one}


