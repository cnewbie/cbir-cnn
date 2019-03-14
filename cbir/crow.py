# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import numpy as np
from .utils import normalize

def compute_crow_spatial_weight(X, a=2, b=2):
    """
    Given a tensor of features, compute spatial weights as normalized total activation.
    Normalization parameters default to values determined experimentally to be most effective.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :param int a:
        the p-norm
    :param int b:
        power normalization
    :returns ndarray:
        a spatial weight matrix of size (height, width)
    """
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return (S / z)**(1./b) if b != 1 else (S / z)


def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros


def apply_crow_aggregation(X):
    """
    Given a tensor of activations, compute the aggregate CroW feature, weighted
    spatially and channel-wise.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        CroW aggregated global image feature    
    """
    S = compute_crow_spatial_weight(X)  # H x W
    C = compute_crow_channel_weight(X)  # C
    X = X * S  # C x H x W * H x W 
    X = X.sum(axis=(1, 2))  # C
    return normalize(np.expand_dims(X * C,axis=0))


def apply_ucrow_aggregation(X):
    """
    Given a tensor of activations, aggregate by sum-pooling without weighting.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        unweighted global image feature    
    """
    return np.expand_dims(X.sum(axis=(1, 2)), axis=0)

def save_spatial_weights_as_jpg(S, path='.', filename='crow_sw', size=None):
    """
    Save an image for visualizing a spatial weighting. Optionally provide path, filename,
    and size. If size is not provided, the size of the spatial map is used. For instance,
    if the spatial map was computed with VGG, setting size=(S.shape[0] * 32, S.shape[1] * 32)
    will scale the spatial weight map back to the size of the image.

    :param ndarray S:
        spatial weight matrix
    :param str path:
    :param str filename:
    :param tuple size:
    """
    img = scipy.misc.toimage(S)
    if size is not None:
        img = img.resize(size)

    img.save(os.path.join(path, '%s.jpg' % str(filename)))