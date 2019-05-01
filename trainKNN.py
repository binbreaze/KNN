import numpy as np
import operator
from KNN import *
from preprocession import *
import os

def trainKNN(filedir):
    redio = 0.1
    trainset , class_label = filetomatrix(filedir)  #加载数据集
    normset , ranger , minset = autonorm(trainset)  #标准化数据
    test_num = trainset.shape[0] * redio            #测试集比例
    input_trainset = normset[test_num-1:,:]         #训练集
    input_trainlabel = class_label[test_num-1:]     #训练集类别
    t_num = 0
    f_num = 0
    for i in range(test_num):
        input_testset = normset[:i,:]
        test_classlabel = KNN(input_testset,input_trainset,input_trainlabel,5)    #对测试集进行分类
        if test_classlabel != class_label[i]:
            print("{test}-----分类错误！！！".format(test=trainset[i]))
            f_num += 1
        else:
            print("{test}------分类正确！！！".format(test=trainset[i]))
            t_num +=1

    acc_redio = t_num / (f_num + t_num)

    print("准确率为：{redio}".format(redio=acc_redio))

def imagetomatrix(filename):
    imset = np.zeros((1,1024))
    with open(filename,"r") as f:
        for i in range(32):
            lines = f.readlines()
            for  j  in range(32):
                imset[0, 32*i + j] = lines[i][j]

    return imset

def train_test_set():
    train_label = []
    train_file = os.listdir("./trainingDigits/")
    trainset = np.zeros((len(train_file),1024))
    for i in range(len(train_file)):
        train_label.append(train_file[1].split('.')[0].split('_')[0])
        trainset[i,:] = imagetomatrix("./trainingDigits/{filename}".format(filename=train_file[i]))

    test_file = os.listdir("./testDigits/")
#    testset = np.zeros((len(test_file),1024))
    test_label = []

    for j in range(len(test_file)):
        test_label.append(test_file[j].split('.')[0].split('_')[0])
        testset = imagetomatrix("./testDigits/{filename}".format(filename=test_file[j]))

    classfiy_result = KNN(testset,trainset,train_label,5)
    print("分类结果为：{result}".format(result=classfiy_result))


