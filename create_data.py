#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2020/11/10 14:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: site_featurizer.py
@Software: Spyder & Python
"""
from pylab import *
import pandas as pd
import numpy as np
import os
import random
import json,pickle
from collections import OrderedDict,defaultdict
import networkx as nx
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import spectral_clustering,affinity_propagation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils import *
from site_featurizer import *

class cluster_targets_seq():
    """
    单个蛋白可能会与多个蛋白建立成对关系，某些可能在训练集中，有些在测试集中，从而导致信息泄漏。
    根据序列相似性进行聚类后，划分数据集
    """
    
    def __init__(self,dpath='../../data/',bsmset_path=None):
        self.dpath = dpath
        self.bsmset_path = bsmset_path
        
    def GetSSWscore(self):
        
        with open(self.dpath + '/Datasets/ssw_score.pkl','rb') as rpklf:
            ssw_score = pickle.load(rpklf)
        iniprot_dict = dict()
        with open(self.dpath + self.bsmset_path) as rpklf:
            for i_data in rpklf:
                i = i_data.rstrip('\n').split('\t')
                if i[0] == 'PDB_A':
                    pass
                else:
                    iniprot_dict[i[3]] = ' '
                    iniprot_dict[i[6]] = ' '
        iniprot_list = list(iniprot_dict.keys())
        iniprot_ij_dict = dict()
        for i in iniprot_list:
            for j in iniprot_list:
                if i == j:
                    pass
                else:
                    iniprot_ij = (sort((i,j))[0],sort((i,j))[1])
                    iniprot_ij_dict[iniprot_ij] = ssw_score[iniprot_ij]
        
        return ssw_score,iniprot_list,iniprot_ij_dict
    
    def MetricsCluster(self):
        # 生成相似性矩阵用于接下来的聚类
        ssw_score,iniprot_list,iniprot_ij_dict = self.GetSSWscore()
        metrics_metrix = list()
        for i in iniprot_list:
            metrics_i = list()
            for j in iniprot_list:
                if i == j:
                    simi = 1
                else:
                    simi = iniprot_ij_dict[(sort((i,j))[0],sort((i,j))[1])]
                metrics_i.append(simi)
            metrics_metrix.append(metrics_i)
        metrics_metrix = np.array(metrics_metrix)
        # 根据相似性矩阵，采用AP近邻传播聚类算法(Frey B.J, Science, 2007;315(5814):972-976)进行聚类
        lables = affinity_propagation(metrics_metrix,preference = 0.3)
        cluster_iniprot =defaultdict(list)
        for i in range(0,len(iniprot_list)):
            cluster_iniprot[lables[1][i]].append(iniprot_list[i])
            
        return cluster_iniprot
            
    def LoadCluster(self):
        cluster_file = self.dpath + 'Datasets/cluster_targets.pkl'
        if ((not os.path.isfile(cluster_file))):
            cluster_targets = self.MetricsCluster()
            with open(cluster_file,'wb') as wpklf:
                pickle.dump(cluster_targets,wpklf)
        else:
            with open(cluster_file,'rb') as rpklf:
                cluster_targets = pickle.load(rpklf)
        
        return cluster_targets
        
        
class split_dataset():
    """
    """
    
    def __init__(self,dpath='../../data/',bsmset_path=None,seed_num=123):
        self.dpath = dpath
        self.bsmset_path = bsmset_path
        self.seed_num = seed_num
        
    def random_pdbs(self,Test,bsmp_label,cluster_targets):

        test_targets = dict()
        for i in Test:
            for j in cluster_targets[i]:
                test_targets[j] = ' '

        test_pdbs = dict()
        for k,v in bsmp_label.items():
            targets = v[1]
            if targets[0] not in test_targets or targets[1] not in test_targets:
                pass
            else:
                test_pdbs[k] = v[0]
        keys = list(test_pdbs.keys())
        r=random.random
        random.seed(123)
        random.shuffle(keys,random=r)
        random.shuffle(keys)
        test_pdbs_random = dict()
        for key in keys:
            test_pdbs_random[key] = test_pdbs[key]
        X_test = list(test_pdbs_random.keys())
        y_test = list(test_pdbs_random.values())
        return X_test,y_test

    def random_train_pdbs(self,Test,bsmp_label,cluster_targets):

        test_targets = dict()
        for i in Test:
            for j in cluster_targets[i]:
                test_targets[j] = ' '

        test_pdbs = dict()
        for k,v in bsmp_label.items():
            targets = v[1]
            if targets[0] not in test_targets and targets[1] not in test_targets:
                pass
            else:
                test_pdbs[k] = v[0]
        keys = list(test_pdbs.keys())
        r=random.random
        random.seed(123)
        random.shuffle(keys,random=r)
        test_pdbs_random = dict()
        for key in keys:
            test_pdbs_random[key] = test_pdbs[key]
        X_test = list(test_pdbs_random.keys())
        y_test = list(test_pdbs_random.values())
        return X_test,y_test

    def LoadData(self):
        bsmp_label = dict()
        targets = dict()
        with open(self.dpath + self.bsmset_path) as rpklf:
            for i_data in rpklf:
                i = i_data.rstrip('\n').split('\t')
                if i[0] == 'PDB_A':
                    pass
                else:
                    bsmp_label[(i[0],i[1])] = (float(i[2]),(i[3],i[6]))
                    targets[i[3]] = ' '
                    targets[i[6]] = ' '

        with open(self.dpath + 'Datasets/cluster_targets.pkl','rb') as rpklf:
            cluster_targets = pickle.load(rpklf)
        cluster_targets_updata = dict()
        for k,v in cluster_targets.items():
            v_updata = list()
            for v_data in v:
                if v_data not in targets:
                    pass
                else:
                    v_updata.append(v_data)
            if v_updata == []:
                pass
            else:
                cluster_targets_updata[k] = v_updata
        cluster_nodes = list(cluster_targets_updata.keys())

        Trains, Valid, _, _, = train_test_split(cluster_nodes, cluster_nodes, test_size = 0.2, 
                                                shuffle=True,random_state=self.seed_num)

        X_train,y_train = self.random_train_pdbs(Trains,bsmp_label,cluster_targets_updata)
        X_valid,y_valid = self.random_pdbs(Valid,bsmp_label,cluster_targets_updata)

        return X_train,y_train,X_valid,y_valid
    
    def SaveSplitSet(self):
        
        X_train,y_train,X_temp,y_temp = self.LoadData() 
        X_valid,X_test,y_valid,y_test = train_test_split(X_temp, y_temp, test_size = 0.5, 
                                                         shuffle=True,random_state=self.seed_num)
        creat_data = [(X_train,y_train),(X_valid,y_valid),(X_test,y_test)]
        datasets = ['BSMset_DelVerBare_train','BSMset_DelVerBare_valid','BSMset_DelVerBare_test']
        for n in range(0,len(datasets)):
            with open(dpath + '/Datasets/' + datasets[n] + '.txt', 'w') as wpklf:
                wpklf.write('PDB_A\tPDB_B\tLabel\n')
                X,y = creat_data[n][0],creat_data[n][1]
                for index in range(0,len(X)):
                    wpklf.write('\t'.join(list(X[index]) + [str(y[index])]) + '\n')
        return 'The dataset was split'
    
    
if __name__ == '__main__':
    """ 
    1. Split the datset; 2. Converting Site to graph
    """
    # Split the datset
    dpath = '/home/tbyang/Desktop/GraphBSM3/data/'
    bsmset_path = 'Datasets/BSMset_DelVerBare.txt'
    seed_num = 1234
    # # 基于序列相似性对靶点进行聚类，产生的聚类结果存放在./data/Datasets/cluster_targets.pkl,无此文件时，将下面两行注释去除
    # CTS = cluster_targets_seq(dpath,bsmset_path)
    # cluster_targets = CTS.LoadCluster()
    SD = split_dataset(dpath,bsmset_path,seed_num)
    logset = SD.SaveSplitSet()
    print(logset)

    # Converting Site to graph
    with open(dpath + 'Datasets/site_graph.pkl','rb') as rpklf:
        site_graph = pickle.load(rpklf)
    datasets = ['BSMset_DelVerBare_train','BSMset_DelVerBare_valid','BSMset_DelVerBare_test']
    # convert to PyTorch data format
    for dataset in datasets:
        processed_data_file = dpath + 'Datasets/processed/' + dataset + '.pt'
        if ((not os.path.isfile(processed_data_file))):
            df = pd.read_csv(dpath + 'Datasets/' + dataset + '.txt', sep='\t', low_memory=False)
            # df = shuffle(df)
            train_drugs, train_prots, train_Y = list(df['PDB_A']), list(df['PDB_B']), list(df['Label'])
            train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)

            # make data PyTorch Geometric ready
            print('preparing ', dataset + '.pt in pytorch format!')
            train_data = TestbedDataset(root=dpath + 'Datasets/', dataset=dataset, xd=train_drugs, xt=train_prots,
                                        y=train_Y, smile_graph=site_graph)
            print(processed_data_file,  ' have been created')
        else:
            print(processed_data_file, ' are already created')