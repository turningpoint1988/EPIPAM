#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemLoss(nn.Module):
    def __init__(self, a1, a2, device):
        super(OhemLoss, self).__init__()
        a1 = torch.tensor(a1)
        self.a1 = a1.to(device)
        a2 = torch.tensor(a2)
        self.a2 = a2.to(device)
        self.criteria = nn.BCELoss()

    def forward(self, logits1, logits2, labels):
        loss1 = self.criteria(logits1, labels)
        loss2 = self.criteria(logits2, labels)
        loss = self.a1*loss1 + self.a2*loss2

        return loss

