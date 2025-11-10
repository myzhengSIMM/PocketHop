# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/7/1 13:25
@author: LiFan Chen
@Filename: ligand_analysis.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
def data_analysis(dir):
    """
    This function is used to analyze positive and negative distribution of a certain ligand.
    :param dir: dir: file path
    :return: file
    """
    file_name = dir.split("/")[-1].split(".")[0].split("_")[0]
    data_list = []
    interaction = {}
    with open(dir,"r") as f_in:
        data_list = f_in.read().strip().split('\n')
    for element in data_list:
        compound = element.split()[0]
        label = element.split()[2]
        if compound in interaction.keys():
            interaction[compound][str(label)] += 1
        else:
            interaction[compound] = {}
            interaction[compound]['0'] = 0
            interaction[compound]['1'] = 0
            interaction[compound][str(label)] += 1

    with open(file_name+"_interaction.csv","w") as f_out:
        f_out.write("{}\n".format(",".join(["compound","positive","negative"])))
        for element in interaction.keys():
            f_out.write("{}\n".format(",".join([element,str(interaction[element]['1']),str(interaction[element]['0'])])))

def count_label(dir):
    file_name = dir.split("/")[-1].split(".")[0].split("_")[0]
    df = pd.read_csv(dir)
    #print(df.iloc[0]['positive'])
    N,_ = df.shape
    num = 0
    with open(file_name + "_different_label.txt", "w") as f_out:
        for i in range(N):
            if df.iloc[i]['positive'] !=0 and df.loc[i]['negative'] != 0:
                num += 1
                f_out.write("{}\n".format(df.iloc[i]['compound']))
    print(num)


def make_dataset(dir,list):
    seed = 1234
    file_name = "GPCR"
    compound_list = []
    data_list = []
    with open(list,"r") as f_in:
        compound_list = f_in.read().strip().split('\n')
    np.random.seed(seed)
    np.random.shuffle(compound_list)
    compound_p = set(compound_list[:500])
    with open(dir,"r") as f_in:
        data_list = f_in.read().strip().split('\n')
    with open("dataset/GPCR_new_test.txt","w") as f_out:
        for element in data_list:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            if compound in compound_p and interaction == '1' :
                f_out.write("{}\n".format(' '.join([compound,protein,str(interaction)])))

    compound_n = set(compound_list[-550:])
    with open(dir, "r") as f_in:
        data_list = f_in.read().strip().split('\n')
    with open("dataset/GPCR_new_test.txt", "a") as f_out:
        for element in data_list:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            if compound in compound_n and interaction == '0':
                f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

    with open("dataset/GPCR_new_train.txt","w") as f_out:
        for element in data_list:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            if (compound in compound_n and interaction == '1') or (compound in compound_p and interaction == '0') or (compound in set(compound_list) and compound not in compound_n and compound not in compound_p):
                f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

def make_external(dir,list):
    compound_list = []
    data_list = []
    with open(list, "r") as f_in:
        compound_list = f_in.read().strip().split('\n')
    with open(dir, "r") as f_in:
        data_list = f_in.read().strip().split('\n')
    n = 700
    i = 0
    j = 0
    with open("dataset/GPCR_external.txt", "w") as f_out:
        for element in data_list:
            if i >= n and j >= n: break
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            if compound not in compound_list and interaction == '1' and i <= n:
                f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))
                i += 1
            if compound not in compound_list and interaction == '0' and j <= n:
                f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))
                j += 1

def make_GPCR_new(dir,list):
    with open(dir, "r") as f_in:
        data_list = f_in.read().strip().split('\n')
    compound_pd = pd.read_csv(list)
    label_dict = {}
    for i in range(len(compound_pd)):
        compound = compound_pd.iloc[i]['compound']
        pos = compound_pd.iloc[i]['positive']
        neg = compound_pd.iloc[i]['negative']
        if pos > 0 and neg > 0:
            label_dict[compound] = {}
            label_dict[compound]['positive'] = pos
            label_dict[compound]['negative'] = neg
    compound_dict = {}
    with open("dataset/GPCR_new.txt","w") as f_out:
        for element in data_list:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            if compound not in label_dict.keys(): continue
            if compound in compound_dict.keys():
                compound_dict[compound][interaction] += 1
                if compound_dict[compound][interaction] <= min(label_dict[compound]['negative'], label_dict[compound]['positive']):
                    f_out.write("{}\n".format(' '.join([compound,protein,str(interaction)])))
            else:
                compound_dict[compound] = {}
                compound_dict[compound]['0'] = 0
                compound_dict[compound]['1'] = 0
                compound_dict[compound][interaction] += 1
                f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

def remove(dir):
    with open(dir, "r") as f_in:
        data_list = f_in.read().strip().split('\n')
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    with open("data/GPCR_new_train_2.txt", "w") as f_out:
        for element in data_list:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dir, ratio=0.8):
    seed = 1234
    np.random.seed(seed)
    with open(dir, "r") as f_in:
        data_list = f_in.read().strip().split('\n')
    np.random.shuffle(data_list)
    n = int(ratio * len(data_list))
    dataset_1, dataset_2 = data_list[:n], data_list[n:]
    with open("data/GPCR_new_train_train.txt", "w") as f_out:
        for element in dataset_1:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

    with open("data/GPCR_new_train_dev.txt", "w") as f_out:
        for element in dataset_2:
            compound = element.split()[0]
            protein = element.split()[1]
            interaction = element.split()[2]
            f_out.write("{}\n".format(' '.join([compound, protein, str(interaction)])))

if __name__ == "__main__":
    dir = "data/GPCR_new_train_2.txt"
    list = "data_analysis/data_different_label.txt"
    # data_analysis(dir)
    # count_label(dir)
    # make_external(dir, list)
    # make_GPCR_new(dir, list)
    # make_dataset(dir, list)
    # remove(dir)
    split_dataset(dir)