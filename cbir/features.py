# -*- coding:utf-8 -*-
import numpy as np

def extract_raw_features(model, image_data):
    """(features is 1 x  H x W x C)
    Extract raw features for a single image.
    return: features shape (1 x H x W x C)
    """
    features = model.predict(image_data)
    return features