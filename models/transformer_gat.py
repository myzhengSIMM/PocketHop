#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2020/12/17 16:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: transformer_gat.py
@Software: Spyder & Python
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from Radam import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# TRANSFORMER GATNet model
class TRANSFORMER_GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=32, n_output=2, num_features_xt=32,
                     n_filters=32, embed_dim=128, output_dim=128, hid_dim=256, dropout=None):
        
        super(TRANSFORMER_GATNet, self).__init__()

        ## graph attention networks layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, hid_dim)
        
        ## transformer encoder laysers
        self.hid_dim = hid_dim
        self.n_layers = 2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim*4, 
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

        ## transformer decoder laysers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim*4, 
                                                        dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)

        ## combined layers
        self.fc1 = nn.Linear(hid_dim, 256)
        # self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(256, n_output)

        ## activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        ## graph input feed-forward
        data_a,data_b = data[0],data[1]
        
        ## load the two pocket graphs
        x, edge_index, batch = data_a.x, data_a.edge_index, data_a.batch
        x_t, edge_index_t, batch_t = data_b.x, data_b.edge_index, data_b.batch
        
        ## applying the GAT Model
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = self.relu(self.fc_g1(x))
        x_t = F.dropout(x_t, p=0.2, training=self.training)
        x_t = F.elu(self.gcn1(x_t, edge_index_t))
        x_t = F.dropout(x_t, p=0.2, training=self.training)
        x_t = self.gcn2(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = self.relu(self.fc_g1(x_t))
        
        ## applying the Transformer Model
        # get the number of different sites and the batch size
        b = data_a.y.shape[0]
        num = np.zeros(b)
        # change [atom total_num, hid dim] --> [batch, atom number, hid dim]
        for k in batch:
            num[k] += 1
        num_t = np.zeros(b)
        for k in batch_t:
            num_t[k] += 1
        all_num = np.hstack((num,num_t))
        max_num = int(all_num.max())
        
        # updata the graph input
        compound = torch.zeros((b,max_num,self.hid_dim),device=self.device)
        for i in range(b):
            if i != 0:
                compound[i,:int(num[i]),:] = x[int(num[i-1]):int(num[i-1])+int(num[i]),:]
            else:
                compound[i, :int(num[i]), :] = x[0:int(num[i]), :]
        del x
        target = torch.zeros((b,max_num,self.hid_dim),device=self.device)
        for i in range(b):
            if i != 0:
                target[i,:int(num_t[i]),:] = x_t[int(num_t[i-1]):int(num_t[i-1])+int(num_t[i]),:]
            else:
                target[i, :int(num_t[i]), :] = x_t[0:int(num_t[i]), :]
        del x_t
        
        # create compound mask 
        mask = torch.ones((b,max_num), device=self.device)
        mask_t = torch.ones((b,max_num), device=self.device)
        for i in range(b):
            mask[i, :int(num[i])] = 0.0
            mask_t[i, :int(num_t[i])] = 0.0
        # mask = [batch, compound length]
        target = target.permute(1,0,2).contiguous()
        mask_t = (mask_t == 1).to(self.device)
        compound = compound.permute(1,0,2).contiguous()
        mask = (mask == 1).to(self.device)
        
        ## transformer input feed-forward
        enc_src = self.encoder(target,src_key_padding_mask=mask_t)
        xc = self.decoder(compound, enc_src,tgt_key_padding_mask=mask, memory_key_padding_mask=mask_t)
        # xc = [batch,max_num, hid_dim]
        xc = xc.permute(1,0,2).contiguous()
        xc = xc[:,0,:]

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.out(xc)

        return out