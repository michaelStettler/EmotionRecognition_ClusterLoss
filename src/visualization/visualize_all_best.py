from visualize_statistics import *
from visualize_activation_maximization import *

from keras.applications.resnet50 import ResNet50
from vis.utils import utils

import numpy as np
from tqdm import tqdm

np.set_printoptions(precision=3, linewidth=200, suppress=True)


def max_activation_all_best(model, n_best, input_img=None):

    images = []
    # loop over all best
    for idx in tqdm(range(np.shape(n_best)[1])):
        l_idx = int(n_best[0, idx] + 1)
        layer_idx = utils.find_layer_idx(model, 'activation_' + str(l_idx))
        filter_idx = int(n_best[1, idx])
        value = n_best[2, idx]

        # get max activation images
        img = max_activation(model, idx, layer_idx, filter_idx, input_img=input_img)

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'L%s_F%s_V%.2f' % (l_idx, filter_idx, value))

        images.append(img)

    return images


if __name__ == '__main__':

    # load data_processing
    activation_values = np.load('../features/activation_values.npy')
    print("loaded activation_values")
    # find all best
    n_best = get_filters_statistics(activation_values, n=24)
    print("get best activation values and index")
    # load model
    model = ResNet50(weights='imagenet', include_top=True)
    print("model loaded")

    images = max_activation_all_best(model, n_best)
    save_stiched_image(images, 'face', 'all_best_1', np.shape(n_best)[1], n_cols=8)

    new_images = max_activation_all_best(model, n_best, input_img=images)
    save_stiched_image(new_images, 'face', 'all_best_2', np.shape(n_best)[1], n_cols=8)
