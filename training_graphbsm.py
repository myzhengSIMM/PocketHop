#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time:     Created on 2021/01/11 14:29
@author:   Tianbiao Yang & Mingyue Zheng
@Email:    Tianbiao_Yang@163.com
@Filename: training_graphbsm.py
@Software: Spyder & Python
"""
import numpy as np
import pandas as pd
import sys, os, random
from tqdm import tqdm
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
import torch.nn.functional as F
        
def SetSeed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    np.random.seed(int(1234))
    
SEED = 1234;
SetSeed(SEED)
  
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

def LoadDataSet(SetName,dpath,BATCH_SIZE):
    
    processed_data_file_valid = dpath + 'Datasets/processed/' + SetName + '.pt'
    processed_data_file_valid_t = dpath + 'Datasets/processed/' + SetName + '_t.pt'
    valid_data = TestbedDataset(root=dpath + 'Datasets/', dataset= SetName)
    valid_data_t = TestbedDataset(root=dpath + 'Datasets/', dataset=SetName + '_t')
    # make data PyTorch mini-batch processing ready
    valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
    valid_loader_t = DataLoader(valid_data_t, batch_size = BATCH_SIZE, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
    VERTEXset = defaultdict(list)
    for index,data in enumerate(valid_loader):
        VERTEXset['PDB_A'].append(data)
    for index,data_t in enumerate(valid_loader_t):
        VERTEXset['PDB_B'].append(data_t)
    
    return VERTEXset['PDB_A'],VERTEXset['PDB_B']


def train(model, device, train_loader,Loss, optimizer, epoch):
    model.train()
    value,value_t = train_loader[0],train_loader[1]
    total_loss = 0
    for n in range(0,len(value)):
        data = (value[n].to(device),value_t[n].to(device))
        # print(torch.tensor(data.y,dtype=torch.int64).view(-1))
        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, torch.as_tensor(data[0].y,dtype=torch.int64).view(-1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(value)


def predicting(model, device, Loss,loader):
    model.eval()
    # total_preds = torch.Tensor()
    # total_labels = torch.Tensor()
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
            tpr, fpr, _ = precision_recall_curve(T, S)
            PRC = auc(fpr, tpr)
        except:
            AUC,PRC = 0,0
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        # total_preds = torch.cat((total_preds, output.cpu()), 0)
        # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        
    return AUC,PRC,total_loss/len(value),recall,precision


def training_gat(hypers):
    ## Hyper-parameter
    modeling,BATCH_SIZE,LR,WEIGHT_DECAY,NUM_EPOCHS,DROPOUT,HID_DIM,SEED,HEADER,CUDA_NAME = hypers
    
    ## Loader dataset
    datasets = ['BSMset_DelVerBare_train','BSMset_DelVerBare_valid','BSMset_DelVerBare_test','Vertex','Barelier']
    dataset_loaders = list()
    for dataset in datasets:
        dpath = '/home/tbyang/Desktop/GraphBSM3/data/'
        dataset_loaders.append(LoadDataSet(dataset,dpath,BATCH_SIZE))
    train_loader, valid_loader, test_loader = dataset_loaders[0],dataset_loaders[1],dataset_loaders[2]
    vertex_loader, barelier_loader = dataset_loaders[3],dataset_loaders[4]

    ## Select the modeling
    # modeling = GATNet
    max_auc, best_epoch_value = 0,0
    model_st = modeling.__name__
    cuda_name = CUDA_NAME
    print('running on ', model_st + '_' + datasets[0])
    Hyper = [model_st,NUM_EPOCHS,BATCH_SIZE,str(LR).split('.')[-1],str(DROPOUT).split('.')[-1],HEADER,HID_DIM] 
    
    ## Training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    # print(device)
    model = modeling(dropout=DROPOUT, hid_dim = HID_DIM, heads = HEADER)
    model.to(device)
    Loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc,best_epoch = 0, -1
    file_AUCs = "./logs/train_" + '_'.join(map(str, Hyper)) + '.logs'
    
    AUC = ('Epoch\tTime(sec)\tLoss_train\tAUC_train\tPRC_train\tAUC_dev\tPRC_dev\tAUC_test\tPRC_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')
    model_file_name = 'model_' + '_'.join(map(str, Hyper))+  '.model'
    # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
    for epoch in range(NUM_EPOCHS):
        loss = train(model, device, train_loader,Loss ,optimizer, epoch+1)
        AUC_train,PRC_train,_,_,_ = predicting(model,device, Loss,train_loader)
        AUC_dev,PRC_dev,LOSS_dev,_,_ = predicting(model,device, Loss,valid_loader)
        AUC_test, PRC_test,_,_,_ = predicting(model,device, Loss,test_loader)
        AUC_vertex, PRC_vertex,_,_,_ = predicting(model,device, Loss,vertex_loader)
        _,_,_,AUC_barelier,PRC_barelier = predicting(model,device,Loss,barelier_loader)
        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch,time,loss,AUC_train,PRC_train,AUC_dev,PRC_dev,AUC_test,PRC_test,AUC_vertex, PRC_vertex,AUC_barelier, PRC_barelier,LOSS_dev]
        # print('\t'.join(map(str, [round(i,4) for i in AUCs])))
        save_AUCs(AUCs,file_AUCs)
        if AUC_dev > max_auc:
            torch.save(model.state_dict(), "./logs/" + model_file_name)
            max_auc = AUC_dev
            re_AUCs = AUCs
        # Early Stop
        epoch_valid_value = AUC_dev
        if epoch_valid_value > best_epoch_value:
            best_epoch_value = epoch_valid_value
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= 15:
            break
    result_AUCs = './logs/result_logs.txt'
    save_AUCs(['_'.join(map(str, Hyper))] + [round(i,4) for i in re_AUCs],result_AUCs)
    return re_AUCs

if __name__ == '__main__':
    start = timeit.default_timer()
    
    ## Hyper-parameter
    BATCH_SIZE = 1024;      print('Batch size: ', BATCH_SIZE)
    LR = 0.001;             print('Learning rate: ', LR)
    DROPOUT = 0.3;          print('Dropout: ', DROPOUT)
    HID_DIM = 512;          print('Hidden dimension: ', HID_DIM)
    HEADER = 16;            print('Header: ', HEADER)
    NUM_EPOCHS = 400;       print('Number of epoch: ', NUM_EPOCHS)
    WEIGHT_DECAY = 0.0001;  print('Weight decay: ', WEIGHT_DECAY)
    CUDA_NAME = "cuda:0";   print('Cuda name:', CUDA_NAME)
    
    modeling = GATNet
    hypers = modeling,BATCH_SIZE,LR,WEIGHT_DECAY,NUM_EPOCHS,DROPOUT,HID_DIM,SEED,HEADER,CUDA_NAME
    re_AUCs = training_gat(hypers)
    
    
