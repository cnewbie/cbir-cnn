
import os 
import numpy as np


def ddt(X):
    pass
    """
    X shape (N x H x W x C)
    """
    if len(X.shape) != 4:
        raise ValueError
    N,H,W,C = X.shape
    K = N * H * W
    X_mean = np.expand_dims(X.sum(axis=(0,1,2))/K,axis=0)  # shape 1 x C
    #print(X_mean.shape)
    X_tmp = X - X_mean
    #print(X_tmp.shape)
    X_tmp = X_tmp.sum(axis=(1,2))
    #print(X_tmp.shape)
    X_cov = np.dot(X_tmp,X_tmp.T)
    #print(X_cov.shape)
    #print(X_cov/(H*W))
    #print(np.cov(X.sum(axis=(1,2))/(H*W)))
    X_eig_val,X_eig_vec = np.linalg.eig(X_cov)
    for i in range(len(X_eig_val)):
        pass
    #print(X_eig_val,X_eig_vec.shape)
    X_cov = X_cov.sum(1)/K
    return X_cov
def ddt2(X):
    pass
    """
    X shape (N x H x W x C)
    """
    if len(X.shape) != 4:
        raise ValueError
    N,H,W,C = X.shape
    K = N * H * W
    X = X.sum(axis=(1,2))/(H * W)
    print(X.shape)
    X_cov = np.cov(X)
    X_eig_val,X_eig_vec = np.linalg.eig(X_cov)
    print(X_eig_val,X_eig_vec.shape)
    
    