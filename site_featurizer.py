#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2020/11/10 14:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: site_featurizer_onehot.py
@Software: Spyder & Python
@Aims:     Change the pockets graph via the one hot
"""
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from sklearn.utils import shuffle
from utils import *

def euclidean_dist(c,dist_threshold):
    """Calculate euclidean distance.
    * Parameters
    ----------
    c: Rank 3 array defining coordinates of nodes in n-euclidean space
    * Returns
    ----------
    adj: Rank 3 tensor defining pairwise adjacency matrix of nodes.
    """
    c = torch.Tensor(c)
    adj_mat = torch.zeros(len(c), len(c)).float()
    for i in range(0, len(c)):
        for j in range(0, len(c)):
            dist = torch.dist(c[i], c[j])
            adj_mat[i, j] = 1 if 0 < dist < dist_threshold else 0
    return adj_mat

def res_features(dpath,pocket_name):
    ''' Parse Protein Pocket Graph
    * Parameters
    ----------
    pocket_name: the pocket name in the Graphs
    * Returns
    ----------
    v: 1. one-hot encoding of aminoacid (length 30),
    c: 1. centered coordinates of aminoacid (x,y,z)
    '''
    data_ = []
    graph_name = dpath + 'Graphs/' + pocket_name + '.txt'
    with open(graph_name, "r") as f:
        for i, _ in enumerate(f):
            data_.append(_)
    v,c = [],[]
    for i, line in enumerate(data_):
        row = [float(j) for j in line.rstrip('\n').split('\t')]
        v.append(row[:-3])
        c.append(row[-3:])
    
    del data_

    v = np.array(v, dtype=float)
    c = np.array(c, dtype=float)
    
    features = torch.Tensor(v).numpy()
    edge_index = torch.nonzero(euclidean_dist(c,6)).numpy()
    res_size = len(features)
    return res_size, features, edge_index

if __name__ == '__main__':
    dpath = '/home/tbyang/Desktop/GraphBSM3/data/'
    pocket_name = [i.split('.txt')[0] for i in os.listdir(dpath + 'Graphs/')]
#     # Apply the following script to generate site graph to save in `dpath + 'Datasets/site_graph.pkl'`
#     site_graph = dict()
#     for pocket in pocket_name:
#         try:
#             g = res_features(dpath,pocket)
#             site_graph[pocket] = g
#         except:
#             print(pocket)
#     with open(dpath + 'Datasets/site_graph.pkl','wb') as wpklf:
#         pickle.dump(site_graph,wpklf)

    # load the pickle of site graph 
    with open(dpath + 'Datasets/site_graph.pkl','rb') as rpklf:
        site_graph = pickle.load(rpklf)

    datasets = ['BSMset','Vertex','Barelier','BSMset_DelVerBare',
                'BSMset_DelVerBare_test','BSMset_DelVerBare_train','BSMset_DelVerBare_valid']
    # convert to PyTorch data format

    for dataset in datasets:
        processed_data_file = dpath + 'Datasets/processed/' + dataset + '.pt'
        if ((not os.path.isfile(processed_data_file))):
            df = pd.read_csv(dpath + 'Datasets/' + dataset + '.txt', sep='\t', low_memory=False)
            df = shuffle(df)
            df.to_csv(dpath + 'Datasets/processed/' + dataset + '_shuf.txt',sep='\t',index=False)
            train_drugs, train_prots, train_Y = list(df['PDB_A']), list(df['PDB_B']), list(df['Label'])
            train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)

            # make data PyTorch Geometric ready
            print('preparing ', dataset + '.pt in pytorch format!')
            train_data = TestbedDataset(root=dpath + 'Datasets/', dataset=dataset, xd=train_drugs, xt=train_prots,
                                        y=train_Y, smile_graph=site_graph)
            print(processed_data_file,  ' have been created')
        else:
            print(processed_data_file, ' are already created')