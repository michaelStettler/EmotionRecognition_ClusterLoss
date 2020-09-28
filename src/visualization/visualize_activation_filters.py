from vis.utils import utils
from scipy.misc import imsave
import numpy as np
import cv2

np.set_printoptions(precision=3, linewidth=200, suppress=True)


# find the max value
def find_max_value(data):
    max_values = 0
    for img in data:
        for layer in img:
            tmp_max = np.max(layer)
            if tmp_max > max_values:
                max_values = tmp_max

    return max_values


# normalize the data by a certain value
def normalize(data, val):
    normalized = np.array(data)
    for img_idx in range(np.shape(data)[0]):
        for layer_idx in range(np.shape(data)[1]):
            normalized[img_idx, layer_idx] = data[img_idx, layer_idx] / val

    return normalized


def normalize_per_filter(data):
    normalized = np.array(data)

    for layer_idx in range(np.shape(data)[1]):
        layer = np.zeros((np.shape(data)[0], np.shape(data[0, layer_idx])[0], np.shape(data[0, layer_idx])[1], np.shape(data[0, layer_idx])[2]))

        # loop over each images in order to add the column to the layer
        for img_idx in range(np.shape(data)[0]):
            layer[img_idx] = data[img_idx, layer_idx]

        # find de max value
        layer_max_val = np.max(layer)

        # loop over each img column again to divide by the max value
        for img_idx in range(np.shape(data)[0]):
            normalized[img_idx, layer_idx] = data[img_idx, layer_idx] / layer_max_val

    return normalized


def visualize_filters(data, name=""):
    for layer_idx in range(np.shape(activation_values)[1]):
    # for layer_idx in [10]:
        # print("layer_idx", layer_idx, "num_filters", np.shape(data[0, layer_idx])[0])
        images = []
        for filter_idx in range(np.shape(data[0, layer_idx])[0]):
            for img_idx in range(np.shape(data)[0]):
                tmp = data[img_idx, layer_idx]
                img = tmp[filter_idx]
                img = img * 255
                # change it to RGB image for stitch method to work
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # add filter_idx on the first images
                if img_idx % np.shape(data)[0] == 0:
                    rgb_img = cv2.putText(rgb_img, "%s" % filter_idx, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, .4, 255)
                images.append(rgb_img)

        stitched = utils.stitch_images(images, cols=np.shape(data)[0])
        imsave("activation_%s_%s.png" % (name, layer_idx), stitched.astype(np.uint8))


if __name__ == "__main__":
    activation_values = np.load("../features/activation_values_rotate_hori.npy")
    print("activation_values shape", activation_values.shape)
    # visualize_filters(activation_values, name="raw")

    # max_value = find_max_value(activation_values)
    # print("max_values", max_value)
    #
    # norm_activation_values = normalize(activation_values, max_value)
    # print(activation_values[0, 30][0])
    # print(norm_activation_values[0, 30][0])
    #
    # visualize_filters(norm_activation_values, name="all_max")

    norm_filt_activation_values = normalize_per_filter(activation_values)
    visualize_filters(norm_filt_activation_values, name="layer_max")
