# -*- coding:utf-8 -*-
import os
import numpy as np
from cbir.utils import normalize


def get_discriminative_fm(X):
    pass
    """
    input shape (N x H x W x C)
    """
    if len(X.shape) == 4:
        sum_X = X.sum(1).sum(1)
    elif len(X.shape) == 2:
        sum_X = X
    else:
        raise ValueError
    #var = sum_X - np.mean(sum_X,axis=0)
    #var = var ** 2
    #var = var.sum(0)

    var = np.var(sum_X,axis=0)
    #print(var.shape)
    indexs = np.argsort(var,axis=0)
    idxs = indexs[::-1]
    return idxs
    #print(indexs)
    #print(var[indexs[:10]])
    
def PWA(X, weights, a=2, b=2):
    """
    X input shape (C x H x W)
    """
    # select_load ='../data/filter_select/select_num_oxford.npy'
    #select_load = '../data/filter_select/select_num_paris.npy'
    #select_num = np.load(select_load)
    #select_num_map=select_num[0:25]
    select_num_map = weights[1:25]
    X=np.array(X)
    if X.shape[0]==1 :#some feature is saved as four dim
        X=X[0]
    aggregated_feature=[]
    # loop all part detectors
    for i, x in enumerate(X):
        # whether select this part detector
        if i in select_num_map:
            # norm
            sum = (x ** a).sum() ** (1. / a)
            if sum != 0:   # 防止分母为零
                weight=(x / sum) ** (1. / b)
            else:
                weight = x

            # weighted sum-polling
            aggregated_feature_part=weight*X
            aggregated_feature_part=aggregated_feature_part.sum(axis=(1, 2))
            aggregated_feature_part_normal=aggregated_feature_part

            # concatenation
            if aggregated_feature==[]:
                aggregated_feature=aggregated_feature_part_normal
            else:
                aggregated_feature=np.row_stack((aggregated_feature,aggregated_feature_part_normal))

    aggregated_feature = aggregated_feature.ravel()
    # norm
    aggregated_feature_normal = normalize(np.array(aggregated_feature), copy=False)
    aggregated_feature_normal=aggregated_feature_normal.reshape((1,-1))
    return aggregated_feature_normal
