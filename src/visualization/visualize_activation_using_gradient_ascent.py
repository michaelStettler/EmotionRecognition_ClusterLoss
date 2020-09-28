'''Visualization of the filters, via gradient ascent in input space.
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

sys.path.insert(0, '../utils/')

from utils import*

# dimensions of the generated pictures for each filter.
img_width = 128  # initially 128
img_height = 128  # initially 128

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'res4a_branch2b'  # res2b_branch2b res2b_branch2c
num_filter_index = 30
single = True
kept_all = True

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def look_filters(num_filter_index, layer_output):
    kept_filters = []
    for filter_index in range(num_filter_index):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d / %d' % (filter_index, num_filter_index))
        start_time = time.time()
        converge = True

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # loss = K.mean(model.output[:, 373])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                converge = False

                if not kept_all:
                    break

        # decode the resulting input image
        if kept_all:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        elif loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

        end_time = time.time()
        if converge:
            print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
        else:
            print('Filter %d processed in %ds -> failed!' % (filter_index, end_time - start_time))

    return kept_filters

def look_layer(layer_dict, layer_name, num_filter_index, n=8):
    """

    :param layer_dict:
    :param layer_name:
    :param num_filter_index:
    :param n: number of columns
    :return:
    """
    print("layer:", layer_name)
    print("num_filter_index:", num_filter_index)

    kept_filters = look_filters(num_filter_index, layer_dict[layer_name].output)
    # print("kept filter len", len(kept_filters))

    # find the number of row we can set in function of the number of filter that has converged
    num_row = len(kept_filters) // n + 1
    print("len(kept_filters)", len(kept_filters))
    print("num_colum", num_row)

    stiched_and_save_filter_to_img(kept_filters, 8, num_row, img_width, img_height, layer_printed_name=layer_name)

    if len(kept_filters) > 64:
        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        n = 8
    elif len(kept_filters) > 49:
        n = 7
    elif len(kept_filters) > 36:
        n = 6
    elif len(kept_filters) > 25:
        n = 5
    else:
        n = 3

    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    stiched_and_save_filter_to_img(kept_filters, n, n, img_width, img_height, layer_printed_name="best_" + layer_name)


# build the network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=False)
model = ResNet50(weights='imagenet', include_top=False)
# model = vgg16.VGG16(include_top=True, weights='imagenet')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

if single:
    # do only one filter
    look_layer(layer_dict, layer_name, num_filter_index)

else:
    # do all filters
    for layer_name in sorted(layer_dict):
        if "bn" in layer_name:
            num_filter_index = layer_dict[layer_name].output_shape[3]
            look_layer(layer_dict, layer_name, num_filter_index)


