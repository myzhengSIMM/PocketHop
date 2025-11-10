#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2020/12/17 14:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: gat.py
@Software: Spyder & Python
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=31, n_output=2, hid_dim=None, heads=None, dropout=None):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads, dropout=0.0)
        self.gcn2 = GATConv(num_features_xd * heads, hid_dim, dropout=0.0)
        self.fc_g1 = nn.Linear(hid_dim, hid_dim)

        # combined layers
        self.fc1 = nn.Linear(hid_dim, int(hid_dim*0.5))
        # self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(int(hid_dim*0.5), n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        
        data_a,data_b = data[0],data[1]
        
        x, edge_index, batch = data_a.x, data_a.edge_index, data_a.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        x_t, edge_index_t, batch_t = data_b.x, data_b.edge_index, data_b.batch
        x_t = F.dropout(x_t, p=0.2, training=self.training)
        x_t = F.elu(self.gcn1(x_t, edge_index_t))
        x_t = F.dropout(x_t, p=0.2, training=self.training)
        x_t = self.gcn2(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = gmp(x_t, batch_t)          # global max pooling
        x_t = self.fc_g1(x_t)
        x_t = self.relu(x_t)

        # sum(x + x_t)
        xc = x + x_t
        # concat
        # xc = torch.cat((x, x_t), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.out(xc)
        return out
