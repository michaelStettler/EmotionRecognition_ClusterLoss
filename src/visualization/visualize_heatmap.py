from keras.applications import VGG16
from keras import activations
from keras.applications.resnet50 import ResNet50
from vis.utils import utils
from vis.visualization import visualize_cam
from vis.visualization import visualize_saliency, overlay
import numpy as np
from scipy.misc import imsave
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys

sys.path.insert(0, '../../models/ResNet/keras/')
from resnet18 import *
from simple import *


weights = 'monkey_2/simple_monkey_2_'
run = '05'
filt_idx = 1


# Build the network with ImageNet weights
# model = VGG16(weights='imagenet', include_top=True)
model = SIMPLE(weights='../models/'+weights+run+'.h5', include_top=True, classes=2)
# model = ResNet50(weights='imagenet', include_top=True)
model.summary()

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
# layer_idx = utils.find_layer_idx(model, 'fc1000')
# layer_idx = utils.find_layer_idx(model, 'activation_49')
layer_idx = -1

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

img_monkey1 = utils.load_img('../../data/raw/Monkey/Neutral_2.png', target_size=(224, 224))
img_monkey2 = utils.load_img('../../data/raw/Monkey/OpenMouthThreat_2.png', target_size=(224, 224))


# for modifier in ['guided', 'relu']:
#     # plt.figure()
#     # f, ax = plt.subplots(1, 2)
#     # plt.suptitle(modifier)
#     for i, img in enumerate([img_monkey1, img_monkey2]):
#         # 373 is the imagenet index corresponding to `macaque`
#         grads = visualize_saliency(model, layer_idx, filter_indices=filt_idx,
#                                    seed_input=img, backprop_modifier=modifier)
#         imsave("results/saliency_%s_%s_%s.png" % ('image_'+str(i), modifier, filt_idx), grads)
#         # ax[i].imshow(grads, cmap='jet')
#
# print("done saliency")


# for modifier in [None, 'guided', 'relu']:
for modifier in [None]:
    # plt.figure()
    # f, ax = plt.subplots(1, 2)
    # plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img_monkey1, img_monkey2]):
        # 373 is the imagenet index corresponding to `macaque`
        grads = visualize_cam(model, layer_idx, filter_indices=filt_idx,
                              seed_input=img, backprop_modifier=modifier)
        # grads = visualize_cam(model, layer_idx, filter_indices=None,
        #                       seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        # print(np.shape(grads))
        # print(np.shape(jet_heatmap))
        imsave("results/heatmap_%s_%s_%s_%s.png" % ('image_'+str(i), modifier, filt_idx, run), overlay(jet_heatmap[:, :, :, 0], img, 0.3))
        # ax[i].imshow(overlay(jet_heatmap[:, :, :, 0], img))

print("done heatmap")
# plt.show()