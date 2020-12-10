#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils import data
from sklearn.metrics import roc_auc_score, average_precision_score

# custom functions defined by user
from EPIattention import DeepEPIAttention
from datasets import EPIDataSetTrain, EPIDataSetTest
from loss import OhemNegLoss, OhemLoss


def test(device, model, state_dict, criterion, test_loader):
        """test the performance of the trained model."""
        # loading model parameters
        model.load_state_dict(state_dict)
        model.to(device)
        criterion.to(device)
        model.eval()
        loss_test = 0.
        label_t_all = []
        label_p_all = []
        for i_batch, sample_batch in enumerate(test_loader):
            enhancer = sample_batch["enhancer"].long().to(device)
            promoter = sample_batch["promoter"].long().to(device)
            label = sample_batch["label"].float().to(device)
            with torch.no_grad():
                label_p, dlabel_p = model(enhancer, promoter)
                loss = criterion(label_p.view(-1), dlabel_p.view(-1), label.view(-1))
            if np.isnan(loss.item()):
                raise ValueError('loss is nan during testing')
            loss_test += loss.item() 
            label_p_all.append(label_p.view(-1).data.cpu().numpy())
            label_t_all.append(label.view(-1).data.cpu().numpy())

        label_t_all = np.array(label_t_all)
        label_p_all = np.array(label_p_all)
        auc = roc_auc_score(label_t_all, label_p_all)
        prauc = average_precision_score(label_t_all, label_p_all)
        # f1 = f1_score(label_t_all, label_p_all > 0.5)
        loss_test /= len(test_loader)

        return auc, prauc


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

    parser.add_argument("-g", dest="gpu", type=str, default='1',
                        help="choose gpu device. eg. '0,1,2' ")
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
    f = open(osp.join(args.checkpoint, '%s_record_inter.txt' % args.name), 'w')
    f.write('fold\ta1\ta2\tAUC\tAUPRC\n')
    # 6-fold cross validation
    for cv in range(8):
        test_chroms = ' '.join(index[cv * 3:(cv + 1) * 3])
        print("The current test data is {}".format(test_chroms))
        # build test data generator
        test_data = EPIDataSetTest(args.data_dir + '/data', args.name, 'inter')
        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # load the pre-trained embedding weights
        embedding_matrix = np.load(osp.join(osp.dirname(__file__), 'embedding_matrix.npy'))
        embedding_weights = torch.from_numpy(embedding_matrix).float()
        # we implement many trials for different weight initialization
        checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
        chk = torch.load(checkpoint_file)
        a1 = chk['a1']
        a2 = chk['a2']
        state_dict = chk['model_state_dict']
        print("cv={}  a1={}  a2={}".format(cv, a1, a2))
        model = DeepEPIAttention(embed_weights=embedding_weights)
        criterion = OhemLoss(a1, a2, device)
        auc, prauc = test(device, model, state_dict, criterion, test_loader)
        f.write("{}\t{}\t{}\t{:.3f}\t{:.3f}\n".format(cv, a1, a2, auc, prauc))
        
    f.close()


if __name__ == "__main__":
    main()

