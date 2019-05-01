import numpy as np
import operator

def KNN(input,dataset,label,k):
    dataset_size0 = dataset.shape[0]
    submatrix = np.sum((dataset - np.tile(input,(dataset_size0,1)))**2,axis=1)**0.5   #计算input与dataset的距离  欧式距离sum((dataset - input(m,1))**2)**2

# 根据距离排序从小到大的排序，返回对应的索引位置
# argsort() 是将x中的元素从小到大排列，返回对应的index（索引）。
    order_index = np.argsort(submatrix)
# 找出k个最近距离中最大的分类

    class_label = {}

    for i in range(k):
        class_l = label[order_index[i]]
        class_label[class_l] = class_label.get(class_l,0) + 1


    sorted_class = sorted(class_label.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class[0][0]


# dict = {}
#
# a = [1,2,2,3,4,5,5,3,1,2,2,2,2,2,3,1,5]
#
# for i in a:
#     dict[i] = dict.setdefault(i,0) + 1
#
# print(dict)
#
# b = sorted(dict.items(),key=operator.itemgetter(1),reverse=True)
# print(b[0][0])