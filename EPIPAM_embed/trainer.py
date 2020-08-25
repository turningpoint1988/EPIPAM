import os
import math
import datetime
import numpy as np
import os.path as osp
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score

import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch,
                 train_loader, test_loader, lr_policy):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.log_headers = [
            'epoch',
            'test/loss',
            'test/auc',
            'test/prauc'
        ]
        with open(osp.join(self.checkpoint, 'log.csv'), 'w') as f:
            f.write('\t'.join(self.log_headers) + '\n')
        self.epoch = 0
        self.auc_best = 0
        self.prauc_best = 0
        self.state_best = None

    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            for i_batch, sample_batch in enumerate(self.train_loader):
                enhancer = sample_batch["enhancer"].long().to(self.device)
                promoter = sample_batch["promoter"].long().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                self.optimizer.zero_grad()
                label_p, dlabel_p = self.model(enhancer, promoter)
                loss = self.criterion(label_p.view(-1), dlabel_p.view(-1), label.view(-1))
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optimizer.step()
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(self.epoch, i_batch,
                                                    loss.item(), self.optimizer.param_groups[0]['lr']))
            # test and save the model with higher accuracy
            auc, prauc, loss_val = self.test()
            # record some key results
            with open(osp.join(self.checkpoint, 'log.csv'), 'a') as f:
                log = [self.epoch] + [loss_val] + [auc] + [prauc]
                log = map(str, log)
                f.write('\t'.join(log) + '\n')

        return self.auc_best, self.prauc_best, self.state_best

    def test(self):
        """test the performance of the trained model."""
        self.model.eval()
        loss_test = 0.
        label_t_all = []
        label_p_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            enhancer = sample_batch["enhancer"].long().to(self.device)
            promoter = sample_batch["promoter"].long().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            with torch.no_grad():
                label_p, dlabel_p = self.model(enhancer, promoter)
                loss = self.criterion(label_p.view(-1), dlabel_p.view(-1), label.view(-1))
            if np.isnan(loss.item()):
                raise ValueError('loss is nan during testing')
            loss_test += loss.item() 
            label_p_all.append(label_p.view(-1).data.cpu().numpy())
            label_t_all.append(label.view(-1).data.cpu().numpy())

        label_t_all = np.array(label_t_all)
        label_p_all = np.array(label_p_all)
        auc = roc_auc_score(label_t_all, label_p_all)
        prauc = average_precision_score(label_t_all, label_p_all)
        loss_test /= len(self.test_loader)
        if (self.prauc_best + self.auc_best) < (prauc + auc):
            self.auc_best = auc
            self.prauc_best = prauc
            self.state_best = deepcopy(self.model.state_dict())
        print("auc: {:.3f}\tprauc: {:.3f}\n".format(auc, prauc))
        return auc, prauc, loss_test
