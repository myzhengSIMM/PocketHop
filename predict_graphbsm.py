#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2020/12/28 14:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: predict_with_pretrained_model.py
@Software: Spyder & Python
"""
import numpy as np
import pandas as pd
import sys, os, pickle, warnings
from collections import defaultdict
from random import shuffle
import torch
import timeit
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.transformer_gat import TRANSFORMER_GATNet
from models.transformer_gcn import TRANSFORMER_GCNNet
from Radam import *
from utils import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, accuracy_score,roc_curve
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def LoadDataSet(SetName,dpath,BATCH_SIZE):
    
    processed_data_file_valid = dpath + 'Datasets/processed/' + SetName + '.pt'
    processed_data_file_valid_t = dpath + 'Datasets/processed/' + SetName + '_t.pt'
    valid_data = TestbedDataset(root=dpath + 'Datasets/', dataset= SetName)
    valid_data_t = TestbedDataset(root=dpath + 'Datasets/', dataset=SetName + '_t')
    # make data PyTorch mini-batch processing ready
    valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle=False)
    valid_loader_t = DataLoader(valid_data_t, batch_size = BATCH_SIZE, shuffle=False)
    VERTEXset = defaultdict(list)
    for index,data in enumerate(valid_loader):
        VERTEXset['PDB_A'].append(data)
    for index,data_t in enumerate(valid_loader_t):
        VERTEXset['PDB_B'].append(data_t)
    
    return VERTEXset['PDB_A'],VERTEXset['PDB_B']

def predicting(model, device, Loss,loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    T, Y, S = [], [], []
    with torch.no_grad():
        value,value_t = loader[0],loader[1]
        total_loss = 0
        for n in range(0,len(value)):
            data = (value[n].to(device),value_t[n].to(device))
            output = model(data)
            loss = Loss(output, torch.as_tensor(data[0].y,dtype=torch.int64).view(-1).to(device))
            total_loss += loss.item()
            correct_labels = torch.as_tensor(data[0].y,dtype=torch.int64).view(-1)
            correct_labels = correct_labels.to('cpu').data.numpy()
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
            #print(T)
            #print(S)
        try:
            AUC = roc_auc_score(T, S)
            fpr_auc, tpr_auc, _ = roc_curve(T, S)
            fpr_prc, tpr_prc, _  = precision_recall_curve(T, S)
            PRC = auc(tpr_prc, fpr_prc)
        except:
            AUC,PRC = 0,0
            fpr_auc, tpr_auc,fpr_prc, tpr_prc = 0,0,0,0
        
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        f1 = f1_score(T, Y, average='weighted')
        acc = accuracy_score(T,Y)
        # total_preds = torch.cat((total_preds, output.cpu()), 0)
        # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        metrics_list = [AUC,PRC,recall,precision,f1,acc]
        fprs_tprs = [fpr_auc, tpr_auc,fpr_prc, tpr_prc]
        TYS = [T,Y,S]
        
    return metrics_list,fprs_tprs,TYS

def predicting_gat(hypers):
    ## Hyper-parameter
    modeling,BATCH_SIZE,LR,WEIGHT_DECAY,NUM_EPOCHS,DROPOUT,HID_DIM,SEED,HEADER,CUDA_NAME = hypers
    
    ## Loader dataset
    datasets = ['BSMset_DelVerBare_train','BSMset_DelVerBare_valid','BSMset_DelVerBare_test','Vertex','Barelier']
    dataset_loaders = list()
    for dataset in datasets:
        dpath = '../../data/'
        dataset_loaders.append(LoadDataSet(dataset,dpath,BATCH_SIZE))
    train_loader, valid_loader, test_loader = dataset_loaders[0],dataset_loaders[1],dataset_loaders[2]
    vertex_loader, barelier_loader = dataset_loaders[3],dataset_loaders[4]

    ## Select the modeling
    # modeling = GATNet
    max_auc = 0
    model_st = modeling.__name__
    cuda_name = CUDA_NAME
    print('running on ', model_st + '_' + datasets[0])
    Hyper = [model_st,NUM_EPOCHS,BATCH_SIZE,str(LR).split('.')[-1],str(DROPOUT).split('.')[-1],HEADER,HID_DIM] 
    model_file_name = 'model_' + '_'.join(map(str, Hyper))+  '.model'
    
    ## Training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)
    model = modeling(dropout=DROPOUT, hid_dim = HID_DIM, heads = HEADER)
    model.to(device)
    Loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc,best_epoch = 0, -1
    
    if os.path.isfile("./logs/" + model_file_name): 
        model.load_state_dict(torch.load("./logs/" + model_file_name))
        metrics_list_train,fprs_tprs_train,TYS_train = predicting(model,device, Loss,train_loader)
        metrics_list_valid,fprs_tprs_valid,TYS_valid = predicting(model,device, Loss,valid_loader)
        metrics_list_test,fprs_tprs_test,TYS_test = predicting(model,device, Loss,test_loader)
        metrics_list_vertex,fprs_tprs_vertex,TYS_vertex = predicting(model,device, Loss,vertex_loader)
        metrics_list_barelier,fprs_tprs_barelier,TYS_barelier = predicting(model,device, Loss,barelier_loader)
        end = timeit.default_timer()
        time = end - start
        AUCs = [[metrics_list_train,fprs_tprs_train,TYS_train],
               [metrics_list_valid,fprs_tprs_valid,TYS_valid],
               [metrics_list_test,fprs_tprs_test,TYS_test],
               [metrics_list_vertex,fprs_tprs_vertex,TYS_vertex],
               [metrics_list_barelier,fprs_tprs_barelier,TYS_barelier]]
        # AUCs = [metrics_list,fprs_tprs,TYS]
    Re_name = 'Re_' + '_'.join(map(str, Hyper))+  '.re'
    with open('../../image/FigS3/S3_B/' + Re_name,'w') as wpklf:
        wpklf.write('\t'.join(map(str, [round(i,4) for i in metrics_list_train])) + '\n')
        wpklf.write('\t'.join(map(str, [round(i,4) for i in metrics_list_valid])) + '\n')
        wpklf.write('\t'.join(map(str, [round(i,4) for i in metrics_list_test])) + '\n')
    print('\t'.join(map(str, [round(i,4) for i in metrics_list_train])))
    print('\t'.join(map(str, [round(i,4) for i in metrics_list_valid])))
    print('\t'.join(map(str, [round(i,4) for i in metrics_list_test]))) 
    
    
    return AUCs

def save_aucs(re_AUCs):
    with open('../../image/Fig05/GraphBSM_test_Re.txt','w') as wpklf:
        for i in range(0,len(re_AUCs[2][2][1])):
            re = [str(re_AUCs[2][2][0][i]),str(round(re_AUCs[2][2][2][i],6))]
            wpklf.write('\t'.join(re) + '\n')
            
    with open('../../image/Fig06/GraphBSM_Vertex_Re.txt','w') as wpklf:
        for i in range(0,len(re_AUCs[3][2][1])):
            re = [str(re_AUCs[3][2][0][i]),str(round(re_AUCs[3][2][2][i],6))]
            wpklf.write('\t'.join(re) + '\n')
                
    with open('../../image/Fig05/Re_AUCs.pkl','wb') as wpklf:
        pickle.dump(re_AUCs,wpklf)
        
    return print('The Task of predicting via pretrained model was finished !!!')
        

if __name__ == '__main__':
    start = timeit.default_timer()
    
    ## Hyper-parameter
    BATCH_SIZE = 1024;     print('Batch size: ', BATCH_SIZE)
    LR = 0.001;            print('Learning rate: ', LR)
    DROPOUT = 0.1;         print('Dropout: ', DROPOUT)
    HID_DIM = 512;         print('Hidden dimension: ', HID_DIM)
    HEADER = 16;            print('Header: ', HEADER)
    SEED = 1234;           print('Seed: ', SEED)
    NUM_EPOCHS = 400;      print('Number of epoch: ', NUM_EPOCHS)
    WEIGHT_DECAY = 0.0001; print('Weight decay: ', WEIGHT_DECAY)
    CUDA_NAME = "cuda:1";  print('Cuda name:', CUDA_NAME)
    
    modeling = GATNet
    torch.manual_seed(SEED)
    hypers = modeling,BATCH_SIZE,LR,WEIGHT_DECAY,NUM_EPOCHS,DROPOUT,HID_DIM,SEED,HEADER,CUDA_NAME
    re_AUCs = predicting_gat(hypers)
    save_aucs(re_AUCs)
