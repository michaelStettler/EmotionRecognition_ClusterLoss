from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras import activations
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from vis.utils import utils
from vis.visualization import get_num_filters
from scipy.misc import imsave
import numpy as np
import sys
import time
import re
from tqdm import tqdm

sys.path.insert(0, '../utils/')

from utils import*

np.set_printoptions(precision=3, linewidth=200, suppress=True)


def dynamic_print(img_idx, layer, filt_idx, num_filters):
    sys.stdout.write("\rimage %s, layer %s, filter %s/%s" % (img_idx, layer, filt_idx, num_filters))
    sys.stdout.flush()
    time.sleep(0.01)


def build_dict_entries(img_idx, layer, filt_idx, n):
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)] = {}
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['max'] = {}
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['%s_max' % n] = {}
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['threshold_max'] = {}


def get_max_value(filt, img_idx, layer, filt_idx):
    # get the max value
    max_ind = np.unravel_index(np.argmax(filt, axis=None), filt.shape)
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['max']['index'] = max_ind
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['max']['values'] = filt[max_ind]


def get_n_max_values(filt, img_idx, layer, filt_idx, n):
    # get the n max values, the maximum values is at the last position
    n_max_ind = np.unravel_index(np.argsort(filt, axis=None)[-n:], filt.shape)
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['%s_max' % n]['index'] = n_max_ind
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['%s_max' % n]['values'] = filt[n_max_ind]


def get_threshold_values(filt, img_idx, layer, filt_idx, threshold):
    # get all the values higher than a threshold
    # get indexes
    thresh_max_ind = np.nonzero(filt > threshold)
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['threshold_max']['index'] = thresh_max_ind
    # get values
    img_dict[str(img_idx)][layer]['filt_' + str(filt_idx)]['threshold_max']['values'] = filt[filt > threshold]


def get_activation(img, layer_name):
    inputs = [K.learning_phase()] + model.inputs
    _activation1_f = K.function(inputs, [layer_dict[layer_name].output])

    def activation_f(X):
        # The [0] is to disable the training phase flag
        return _activation1_f([0] + [X])

    return activation_f(img)


def get_neuron_activation(layer, filter_idx, row, col, dict, images):
    neuron_dict[layer]['filter_' + str(filter_idx)] = {}
    neuron_dict[layer]['filter_' + str(filter_idx)]['row_' + str(row) + '_col_' + str(col)] = []

    for idx, img in enumerate(images):
        print("Activation output")

        A1 = get_activation(img, layer)
        print("np.shape A1", np.shape(A1[0]))
        print("np.shape A1", np.shape(A1[0][0]))
        print(A1[0][0][:, :, filter_idx])
        filter = A1[0][0][:, :, filter_idx]
        max_ind = np.unravel_index(np.argmax(filter, axis=None), filter.shape)
        print("max_index", max_ind, filter[max_ind])

        # store neuron with image activation
        dict[layer]['filter_' + str(filter_idx)]['row_' + str(row) + '_col_' + str(col)].append(A1[0][0][row, col, filter_idx])

    return dict


def get_image_activation(dict, images, layer_dict):
    n = 3
    threshold = 2

    activation_values = []
    # for each images
    for img_idx, img in enumerate(tqdm(images)):
        # add img to img_dict
        img_dict[str(img_idx)] = {}

        layer_idx = 0
        activation_values.append([])
        # print("\nimg_idx\n", idx)
        # for each activation layers
        for layer in (x for x in tqdm(layer_dict) if re.match("activation", x)):
        # layer_name = 'activation_1'
        # for layer in (x for x in layer_dict if x == layer_name):
            # add the layer to the dict
            img_dict[str(img_idx)][layer] = {}
            # expand activation array
            activation_values[img_idx].append([])

            # retrieve info from the model
            model_layer_idx = utils.find_layer_idx(model, layer)
            num_filters = get_num_filters(model.layers[model_layer_idx])
            A1 = get_activation(img, layer)

            #loop over the filter in the layer
            for filt_idx in range(num_filters):
            # for filt_idx in [0, 1, 2]:
            #     dynamic_print(img_idx, layer, filt_idx, num_filters)
                # add the filter idx to the fictionaries, each filter store the max, the n_max and the threshold values
                build_dict_entries(img_idx, layer, filt_idx, n)

                # get current filter
                filt = A1[0][0][:, :, filt_idx]
                activation_values[img_idx][layer_idx].append(filt)

                # get max, n-max and threshold values
                get_max_value(filt, img_idx, layer, filt_idx)
                get_n_max_values(filt, img_idx, layer, filt_idx, n)
                get_threshold_values(filt, img_idx, layer, filt_idx, threshold)

            layer_idx = layer_idx + 1
    return dict, activation_values


if __name__ == "__main__":
    start_time = time.time()
    # Build the network with ImageNet weights
    model = ResNet50(weights='imagenet', include_top=True)
    # model.summary()

    images = []
    # raw_img = ['results/macaque_373_1.png', 'results/macaque_373_2.png']
    path_folder = import_img_name_from_files("../../data_processing/processed/Maya/Face/face_invariant")
    paths = []
    for path_img in sorted(path_folder):
        x = image.load_img(path_img, target_size=(224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # layer_name = 'activation_49'
    # layer_name = 'activation_39'
    # print("layer_name:", layer_name)
    # print(layer_dict[layer_name].get_config())
    # print(layer_dict[layer_name].get_weights())

    # neuron_dict = {}
    # neuron_dict[layer_name] = {}

    img_dict = {}

    # filter_idx = 0
    # row_idx = 0
    # col_idx = 0
    # neuron_dict = get_neuron_activation(layer=layer_name,
    #                                     filter_idx=filter_idx,
    #                                     row=row_idx,
    #                                     col=col_idx,
    #                                     dict=neuron_dict,
    #                                     images=images)
    # print(neuron_dict)

    img_dict, activation_values = get_image_activation(img_dict, images, layer_dict)
    # # # print(img_dict)
    # # save_obj(img_dict, 'horizontal_' + layer_name + '_dict')
    np.save("activation_values", activation_values)
    print("activation values' shape", np.shape(activation_values))
    print("\ntotal time:", time.time() - start_time)
