# -*- coding:utf-8 -*-
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from .utils import normalize

def compute_distances(X, Y, names,metric='euclidean'):
    """
    :params X (1 x C)
    :params Y (N x C)
    :params metric euclidean, cosine, l1, l2
    return idxs, rank_dists, rank_names
    """
    dists = np.squeeze(pairwise_distances(X, Y,metric=metric))
    idxs = np.squeeze(np.argsort(dists)) 
    rank_dists = dists[idxs]
    rank_names = [names[i] for i in idxs]
    return idxs, rank_dists, rank_names
    
def query():
    pass

def compute_cosin_distance(Q, feats, names):
    """
    feats and Q: L2-normalize, n*d
    """
    dists = np.squeeze(np.dot(Q, feats.T))
    #print(dists.shape)
    idxs = np.argsort(dists)[::-1]
    #idxs = np.argsort(dists)
    rank_dists = dists[idxs]
    rank_names = [names[k] for k in idxs]
    return (idxs, rank_dists, rank_names)

def simple_query_expansion(Q, data, inds, top_k=10):
    """
    Get the top-k closest vectors, average and re-query
    :param ndarray Q:
        query vector
    :param ndarray data:
        index data vectors
    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param int top_k:
        the number of closest vectors to consider
    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    #Q += data[inds[:top_k], :].sum(axis=0)
       
    # weighted query
    for i in range(top_k):
        Q += (1.0*(top_k-i)/float(top_k))*data[inds[i], :]
    return normalize(Q)


def reranking(Q, data, inds, names, top_k = 50):
    vecs_sum = data[0, :]
    for i in range(1, top_k):
        vecs_sum += data[inds[i], :]
    vec_mean = vecs_sum/float(top_k)
    Q = normalize(Q - vec_mean)
    for i in range(top_k):
        data[i, :] = normalize(data[i, :] - vec_mean)
    sub_data = data[:top_k]
    sub_idxs, sub_rerank_dists, sub_rerank_names = compute_distances(Q, sub_data, names[:top_k])
    names[:top_k] = sub_rerank_names
    return names
