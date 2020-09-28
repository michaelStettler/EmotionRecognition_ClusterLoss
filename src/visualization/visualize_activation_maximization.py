from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras import activations
from vis.utils import utils
from vis.input_modifiers import Jitter
from vis.visualization import visualize_activation
from vis.visualization import get_num_filters
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np
from tqdm import tqdm


def save_stiched_image(data, name, layer_name, n_filters, n_cols=8):
    # Generate stitched image palette with n columns
    stitched1 = utils.stitch_images(data, cols=n_cols)
    imsave("results/max_activation/max_activation_%s_%s_%s.png" % (layer_name, name, n_filters), stitched1)
    np.save('results/np_images/max_activation_%s_%s_%s' % (layer_name, name, n_filters), data)


def max_activation_last_layer(model, filter_indices=373):
    # see what a macaque look like
    # 20 is the imagenet category for 'ouzel'
    # 373 is the imagenet category for macaque

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    # layer_idx = utils.find_layer_idx(model, 'fc1000')
    layer_idx = -1

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    img = visualize_activation(model, layer_idx, filter_indices=filter_indices, max_iter=500, verbose=True, input_modifiers=[Jitter(16)])
    # plt.imshow(img)
    imsave("macaque_373.png", img)


def max_activation(model, idx, layer_idx, filter_idx, input_img):
    if input_img is None:
        img = visualize_activation(model, layer_idx,
                                   filter_indices=filter_idx,
                                   tv_weight=0.,
                                   input_modifiers=[Jitter(0.05)])
    else:
        # We will seed with optimized image (input) this time.
        img = visualize_activation(model, layer_idx,
                                   filter_indices=filter_idx,
                                   seed_input=input_img[idx],
                                   input_modifiers=[Jitter(0.05)])
    return img


def max_activation_layers(model, layer_name, selected_filter, input_img=None):
    print("layer_name:", layer_name)
    layer_idx = utils.find_layer_idx(model, layer_name)

    print("len filters", len(selected_filter))
    # Generate input image for each filter.
    images = []
    for idx, filter_idx in enumerate(tqdm(selected_filter)):
        img = max_activation(model, idx, layer_idx, filter_idx, input_img)

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(filter_idx))
        images.append(img)

    return images


if __name__ == '__main__':
    # Build the network with ImageNet weights
    model = ResNet50(weights='imagenet', include_top=True)
    # model.summary()

    # %matplotlib inline
    plt.rcParams['figure.figsize'] = (18, 6)

    # for layer_name in ['activation_5', 'activation_11', 'activation_21', 'activation_23', 'activation_39', 'activation_41']:
    # for layer_name in ['bn2a_branch2a', 'bn2a_branch2b', 'bn2a_branch2c', 'bn2a_branch1']:
    # for layer_name in ['bn5b_branch2a', 'bn5b_branch2b', 'bn5b_branch2c', 'bn5c_branch2a', 'bn5c_branch2b', 'bn5c_branch2c']:
    #
    # to do
    # bn_conv1
    # bn2b_branch2a bn2b_branch2b bn2b_branch2c
    # bn2c_branch2a bn2c_branch2b bn2c_branch2c
    # bn3a_branch2b bn3a_branch2c bn3a_branch1
    # bn3b_branch2a bn3b_branch2b bn3b_branch2c
    # bn3c_branch2a bn3c_branch2b bn3c_branch2c
    # bn3d_branch2a bn3d_branch2b bn3d_branch2c
    # bn4a_branch2a bn4a_branch2b bn4a_branch2c bn4a_branch1
    # bn4b_branch2a bn4b_branch2b bn4b_branch2c
    # bn4c_branch2a bn4c_branch2b bn4c_branch2c
    # bn4d_branch2a bn4d_branch2b bn4d_branch2c
    # bn4e_branch2a bn4e_branch2b bn4e_branch2c
    # bn4f_branch2a bn4f_branch2b bn4f_branch2c
    # bn5a_branch2a bn5a_branch2b bn5a_branch2c bn5a_branch1
    # bn5b_branch2a bn5b_branch2b bn5b_branch2c
    # bn5c_branch2a bn5c_branch2b bn5c_branch2c
    layer_name = 'activation_18'

    # done
    # bn2a_branch2a bn2a_branch2b bn2a_branch2c bn2a_branch1
    # bn2b_branch2a
    # bn3a_branch2a
    # bn3d_branch2b
    # bn4a_branch2a
    # bn4f_branch2b
    # bn5a_branch2a

    # filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    # filters = np.arange(16)
    # filters = [1136, 1170, 1968, 175, 1034, 281, 1747, 1604]  # layer 48 (activation_49)
    # filters = [71, 236, 186, 157, 232, 75, 45, 161]  # layer 38 (activation_39)
    # filters = [64, 94, 26, 116, 7, 89, 107, 70]  # layer 17 (activation_18)
    # filters = np.arange(get_num_filters(model.layers[layer_idx]))  # put within the arange function to avoid modifying the rest of the code
    filters = [64, 94, 26, 116, 7, 89, 107, 70]  # layer 17 (activation_18)

    vis_images = max_activation_layers(model, layer_name, filters)
    save_stiched_image(vis_images, 'filter_1', layer_name, len(filters))

    new_vis_images = max_activation_layers(model, layer_name, filters, input_img=vis_images)
    save_stiched_image(new_vis_images, 'filter_2', layer_name, len(filters))

