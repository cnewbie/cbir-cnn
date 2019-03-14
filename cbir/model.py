# -*- coding:utf-8 -*-
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


def get_model(model_name='vgg16', pooling=None, layer_name='block5_pool'):
    
    if model_name == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=False, pooling=pooling)
    else:
        model = VGG16(weights='imagenet', include_top=False, pooling=pooling)
    layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    return layer_model