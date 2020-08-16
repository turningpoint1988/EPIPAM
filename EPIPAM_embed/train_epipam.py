#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from EPIattention import DeepEPIAttention
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemLoss
from utils import get_n_params


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="EPIAttention Network for EPI interaction identification")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-b", dest="batch_size", type=int, default=1,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Base learning rate.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=1,
                        help="Number of training steps.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    index = ['chr'+str(i+1) for i in range(23)]
    index[22] = 'chrX'
    f = open(osp.join(args.checkpoint, '%s_record.txt' % args.name), 'w')
    f.write('fold\ta1\ta2\tAUC\tAUPRC\n')
    # 6-fold cross validation
    for cv in range(8):
        test_chroms = ' '.join(index[cv * 3:(cv + 1) * 3])
        print("The current test data is {}".format(test_chroms))
        # build training data generator
        train_data = EPIDataSetTrain(args.data_dir + '/data', args.name, cv)
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
        # build test data generator
        test_data = EPIDataSetTest(args.data_dir + '/data', args.name, cv)
        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # load the pre-trained embedding weights
        embedding_matrix = np.load(osp.join(osp.dirname(__file__), 'embedding_matrix.npy'))
        embedding_weights = torch.from_numpy(embedding_matrix).float()
        # we implement many trials for different weight initialization
        auc_best = 0; prauc_best = 0; a1_best = 0; a2_best = 0
        a1_set = [1., 0.8, 0.6, 0.4, 0.2, 0.]
        a2_set = [0., 0.2, 0.4, 0.6, 0.8, 1.]
        for a1, a2 in zip(a1_set, a2_set):
            print("cv={}  a1={}  a2={}".format(cv, a1, a2))
            model = DeepEPIAttention(embedding_weights)
            total_params = get_n_params(model.parameters())
            print(f'Num params: {total_params:,}')
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
            criterion = OhemLoss(a1, a2, device)
            start_epoch = 0

            # if there exists multiple GPUs, using DataParallel
            if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

            executor = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               device=device,
                               checkpoint=args.checkpoint,
                               start_epoch=start_epoch,
                               max_epoch=args.max_epoch,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               lr_policy=None)

            auc, prauc, state_dict = executor.train()
            if (auc_best + prauc_best) < (auc + prauc):
                auc_best = auc
                prauc_best = prauc
                a1_best = a1
                a2_best = a2
                checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
                torch.save({
                    'a1': a1,
                    'a2': a2,
                    'model_state_dict': state_dict
                }, checkpoint_file)
        f.write("{}\t{}\t{}\t{:.3f}\t{:.3f}\n".format(cv, a1_best, a2_best, auc_best, prauc_best))
        f.flush()
    f.close()


if __name__ == "__main__":
    main()

