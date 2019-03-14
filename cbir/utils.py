# -*- coding:utf-8 -*-
import os
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

def load_image(img_path, img_size=(224,224)):
    """(image date is 1 x  H x W x C)
    Extract raw features for a single image.
    """
    
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def covert_data_format(X, format_type=1):
    """
    input: X shape (1 x H x W x C)
    return: X shape (C x H x W)
    """
    if len(X.shape) == 4 and X.shape[0] == 1:
        X = np.squeeze(X)
    if K.image_data_format() == 'channels_last':  # H x W x C
        pass
        X = X.transpose(-1,-3,-2)
    else:
        pass
    return X
        
def get_list(path, file_type='jpg'):
    """
    Returns a list of filenames for
    all files in a directory. 
    """
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(file_type)])

def load_npy_files(files):
    """
    Function : load features from npy files
    files : list of files
    
    return:
        fests:  [num_samples,num_shapes]
        names:  list of file name
    """
    
    feats = np.vstack(np.load(i) for i in files)
    names = [os.path.splitext(os.path.basename(i))[0] for i in files]
    if len(names) == feats.shape[0]:
        print('successful loading of npy file !!!')
    return feats, names

def normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy=copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]
        
def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
    """
    Given a set of feature vectors, process them with PCA/whitening and return the transformed features.
    If the params argument is not provided, the transformation is fitted to the data.

    :param ndarray features:
        image features for transformation with samples on the rows and features on the columns
    :param int d:
        dimension of final features
    :param bool whiten:
        flag to indicate whether features should be whitened
    :param bool copy:
        flag to indicate whether features should be copied for transformed in place
    :param dict params:
        a dict of transformation parameters; if present they will be used to transform the features

    :returns ndarray: transformed features
    :returns dict: transform parameters
    """
    # Normalize
    features = normalize(features, copy=copy)

    # Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=copy)
        features = pca.fit_transform(features)
        params = { 'pca': pca }

    # Normalize
    features = normalize(features, copy=copy)

    return features, params

def result_precision(postive_set, rank_list):
    pass
    PN = len(postive_set)
    
    precision = 0
    k = 0
    for i,f in enumerate(rank_list):
        if f in postive_set:
            k = k + 1
            precision += k / (i+1)
            #print(precision,k,i)
    result = precision/PN
    return result

def result_recall(postive_set, negative_set, rank_list):
    pass
    PN = len(postive_set)
    
    recall = 0
    k = 0
    for i,f in enumerate(rank_list):
        if f in postive_set:
            k = k + 1
            recall = k / PN
            #print(k,recall)
    return recall

def result_ap(postive_set, negative_set, rank_list):
    pass
    """
    Error compute equation
    """
    PN = len(postive_set)
    RN = len(rank_list)
    NN = len(negative_set)
    
    intersect = 0
    recall = 0
    precision = 1
    ap = 0
    k = 0
    for i in rank_list:
        if i in postive_set:
            intersect = intersect + 1
        if i in negative_set:
            pass
        recall_tmp = intersect / PN
        precision_tmp = intersect / (k  + 1)
        ap += (recall_tmp - recall)*((precision + precision_tmp)/2)
        recall = recall_tmp
        precision = precision_tmp
        k = k + 1
     
    return ap

def compute_ap(pos_set,rank_list):
    """
        implement ap function 
        from oxford c++
    """
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    for i in range(len(rank_list)):
        if rank_list[i] in pos_set:
            intersect_size += 1
            recall = intersect_size / len(pos_set)
            precision = intersect_size / (i+1)
            ap += (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision

    return ap

def get_list_set(query_name, files_dict, delimiter = '_'):
    pass
    #print(query_name)
    postive_set = set()
    negative_set = set()
    for k,v in files_dict.items():
        if k.split(delimiter)[0] == query_name:
            postive_set.add(k)
        else:
            negative_set.add(k)
    
    return postive_set, negative_set
