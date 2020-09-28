'''Visualization of the filters via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

import numpy as np
import time
from scipy.misc import imsave
from keras.applications import vgg16
from keras.applications.resnet50 import ResNet50
from keras import backend as K

# dimensions of the generated pictures for each filter.
img_width = 224
img_height = 224

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'output'
filter_index = 373  # macaque : 373
# util function to convert a tensor into a valid image


# build the network with ImageNet weights
model = ResNet50(weights='imagenet', include_top=True)
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


#visualize the last layer
loss = K.mean(model.output[:, filter_index])
# layer_output = layer_dict['res2a_branch2c'].output
# loss = K.mean(layer_output[:, :, :, 0])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads = normalize(grads)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

# step size for gradient ascent
step = 1.

# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128

# run gradient ascent for 20 steps
for i in range(250):
    loss_value, grads_value = iterate([input_img_data])
    print(i, " loss_value", loss_value)
    input_img_data += grads_value * step


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
