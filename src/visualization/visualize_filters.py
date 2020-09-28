'''Visualization of the filters values
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''

from __future__ import print_function
import numpy as np
import time
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from scipy.misc import imsave
import sys
import re

sys.path.insert(0, '../utils/')

from utils import*

single = True

# build the network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=False)
model = ResNet50(weights='imagenet', include_top=False)
# model = vgg16.VGG16(include_top=True, weights='imagenet')
print('Model loaded.')
layer_name = "activation_1"  # conv1

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

if single:
    # take care of one layer
    layers = [layer_dict[layer_name].get_weights()[0][:, :, :, i] for i in
              range(np.shape(layer_dict[layer_name].get_weights()[0])[3])]

    filter_width = np.shape(layer_dict[layer_name].get_weights()[0])[0]
    filter_height = np.shape(layer_dict[layer_name].get_weights()[0])[1]

    num_column = 8
    num_row = np.shape(layer_dict[layer_name].get_weights()[0])[3] // num_column
    print("num_row", num_row)
    stiched_and_save_filter_to_img(layers, num_column, num_row, filter_width, filter_height,
                                   layer_printed_name=layer_name+"_values")
else:
    #take care of other conv layers
    # note it does not take care of conv1
    for layer_name in sorted(layer_dict):
        if re.match("res\w\w_branch2b", layer_name):
            print("layer_name", layer_name)
            print(np.shape(layer_dict[layer_name].get_weights()[0]))
            layers = [layer_dict[layer_name].get_weights()[0][:, :, i, 0] for i in range(np.shape(layer_dict[layer_name].get_weights()[0])[2])]

            print(np.shape(layers))

            filter_width = np.shape(layer_dict[layer_name].get_weights()[0])[0]
            filter_height = np.shape(layer_dict[layer_name].get_weights()[0])[1]

            print("filter_width", filter_width, "filter_height", filter_height)

            num_column = 8
            num_row = np.shape(layer_dict[layer_name].get_weights()[0])[3] // num_column
            print("num_row", num_row)
            stiched_and_save_filter_to_img(layers, num_column, num_row, filter_width, filter_height, layer_name=layer_name+"_values")
