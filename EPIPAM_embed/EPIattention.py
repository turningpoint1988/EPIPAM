# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import sys, os


class DeepEPIAttention(nn.Module):
    def __init__(self, embed_weights=None):
        super(DeepEPIAttention, self).__init__()
        # embedding layer
        if embed_weights is None:
            print("Embedding matrix is not existed")
            sys.exit(0)
        self.embed_layer_en = nn.Embedding.from_pretrained(embed_weights, freeze=False)
        self.embed_layer_pr = nn.Embedding.from_pretrained(embed_weights, freeze=False)
        # for enhancer
        self.enhancer_conv1 = nn.Conv1d(in_channels=100, out_channels=64, kernel_size=25)
        self.enhancer_pool1 = nn.MaxPool1d(kernel_size=10, stride=10)
        self.enhancer_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=13)
        self.enhancer_pool2 = nn.MaxPool1d(kernel_size=10, stride=10)
        self.enhancer_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7)
        self.enhancer_pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        # for promoter
        self.promoter_conv1 = nn.Conv1d(in_channels=100, out_channels=64, kernel_size=25)
        self.promoter_pool1 = nn.MaxPool1d(kernel_size=10, stride=10)
        self.promoter_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=13)
        self.promoter_pool2 = nn.MaxPool1d(kernel_size=10, stride=10)
        self.promoter_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7)
        self.promoter_pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        # for 1-th attention module
        self.pam1 = SPAM(ratio=0.01)
        # for 2-th attention module
        self.pam2 = SPAM(ratio=0.1)
        # for 3-th attention module
        self.pam3 = SPAM(ratio=1)
        # for merging stage1
        c_in_1 = 384 # 640 (8/4/4)
        self.merge_bn1 = nn.BatchNorm1d(num_features=c_in_1)
        self.linear1 = nn.Linear(c_in_1, 64) 
        self.out_layer1 = nn.Linear(64, 1) 
        # for merging stage1
        c_in_2 = 643 # 245 (8/4/4)
        self.merge_bn2 = nn.BatchNorm1d(num_features=c_in_2)
        self.linear2 = nn.Linear(c_in_2, 64)
        self.out_layer2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.merge_drop = nn.Dropout(p=0.5)
        self.linear_drop = nn.Dropout(p=0.5)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, enhancer, promoter):
        """Construct a new computation graph at each froward"""
        b, _ = enhancer.size()
        # embedding branch
        enhancer_embed = self.embed_layer_en(enhancer)
        enhancer_embed = enhancer_embed.permute(0, 2, 1)
        promoter_embed = self.embed_layer_pr(promoter)
        promoter_embed = promoter_embed.permute(0, 2, 1)
        # the 1-th enhancer branch
        out_e = self.enhancer_conv1(enhancer_embed)
        out_e = self.relu(out_e)
        out_e = self.enhancer_pool1(out_e)
        out_e = self.dropout(out_e)
        # the 1-th promoter branch
        out_p = self.promoter_conv1(promoter_embed)
        out_p = self.relu(out_p)
        out_p = self.promoter_pool1(out_p)
        out_p = self.dropout(out_p)
        # the 1-th interaction
        out_e_1, out_p_1, out_ep_1 = self.pam1(out_e, out_p)
        # the 2-th enhancer branch
        out_e = self.enhancer_conv2(out_e_1)
        out_e = self.relu(out_e)
        out_e = self.enhancer_pool2(out_e)
        out_e = self.dropout(out_e)
        # the 2-th promoter branch
        out_p = self.promoter_conv2(out_p_1)
        out_p = self.relu(out_p)
        out_p = self.promoter_pool2(out_p)
        out_p = self.dropout(out_p)
        # the 2-th interaction
        out_e_2, out_p_2, out_ep_2 = self.pam2(out_e, out_p)
        # the 3-th enhancer branch
        out_e = self.enhancer_conv3(out_e_2)
        out_e = self.relu(out_e)
        out_e = self.enhancer_pool3(out_e)
        out_e = self.dropout(out_e)
        # the 3-th promoter branch
        out_p = self.promoter_conv3(out_p_2)
        out_p = self.relu(out_p)
        out_p = self.promoter_pool3(out_p)
        out_p = self.dropout(out_p)
        # the 3-th interaction
        out_e_3, out_p_3, out_ep_3 = self.pam3(out_e, out_p)
        # merging stage1
        out_merge1 = torch.cat((out_e_3.view(b, -1), out_p_3.view(b, -1)), dim=-1)
        out_merge1 = self.merge_bn1(out_merge1)
        out_merge1 = self.merge_drop(out_merge1)
        out_merge1 = self.linear1(out_merge1)
        out_merge1 = self.relu(out_merge1)
        out_merge1 = self.linear_drop(out_merge1)
        out_merge1 = self.out_layer1(out_merge1)
        out_merge1 = self.sigmoid(out_merge1)
        # merging stage2
        out_merge2 = torch.cat((out_ep_1, out_ep_2, out_ep_3), dim=-1)
        out_merge2 = self.merge_bn2(out_merge2)
        out_merge2 = self.merge_drop(out_merge2)
        out_merge2 = self.linear2(out_merge2)
        out_merge2 = self.relu(out_merge2)
        out_merge2 = self.linear_drop(out_merge2)
        out_merge2 = self.out_layer2(out_merge2)
        out_merge2 = self.sigmoid(out_merge2)

        return out_merge1, out_merge2


class SPAM(nn.Module):
    """simple position attention module"""
    def __init__(self, ratio=0.01):
        super(SPAM, self).__init__()
        self.ratio = ratio

    def forward(self, e, p):
        e_b, _, _ = e.size()
        p_b, _, _ = p.size()
        assert e_b == p_b
        e_t = e.permute(0, 2, 1)
        attention_matrix = torch.bmm(e_t, p) / e_t.size(-1)
        
        b, c, l = attention_matrix.size()
        out_ep, _ = torch.topk(attention_matrix.view(b, -1), int(c * l * self.ratio), dim=1, sorted=False)
        
        attention_e, _ = torch.max(attention_matrix, dim=2)
        attention_e = F.softmax(attention_e.view(e_b, 1, -1), dim=2)
        out_e = attention_e * e + e
        attention_p, _ = torch.max(attention_matrix, dim=1)
        attention_p = F.softmax(attention_p.view(p_b, 1, -1), dim=2)
        out_p = attention_p * p + p

        return out_e, out_p, out_ep


