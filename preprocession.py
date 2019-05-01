# coding: utf-8
import numpy as np
from numpy import zeros
from operator import *

def filetomatrix(filedir):  #读取数据，构造数据集
    file = open(filedir)
    num_lines = len(file.readlines())
    trainset = zeros((num_lines,3))
    class_label = []
    with open(filedir,"r") as f:
        index = 0
        for fileline in f.readlines():
            filelist = fileline.strip().split("\t")
            trainset[index,:] = filelist[0:3]
            class_label.append(int(filelist[-1]))
            index += 1
    return trainset,class_label   #返回训练集与分类


def autonorm(dataset):    #数据归一化处理  dataset - min / max - min
    maxset = dataset.max(0)
    minset = dataset.min(0)
    print(len(dataset))
    m0 = dataset.shape[0]
    fenzi = dataset - np.tile(minset,(m0,1))
    ranger_set  = maxset - minset
    print(len(ranger_set))
    normldataset = fenzi / ranger_set
    print(len(normldataset))
    return normldataset,ranger_set,minset
dataset,class_lable = filetomatrix("./datingTestSet2.txt")
print(autonorm(dataset))


