from __future__ import print_function
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=200, suppress=True)


def get_statistics_per_img(values, img_idx=0):
    layers_mean = []
    layers_max = []
    layers_median = []

    # loop over each layer
    for layer_idx in range(np.shape(values)[1]):
        layer = values[img_idx, layer_idx]

        layers_mean.append(np.mean(layer))
        layers_max.append(np.max(layer))
        layers_median.append(np.median(layer))

        # print("np.shape", np.shape(layer), "argmax", np.amax(layer))
        # print(np.unravel_index(np.argmax(layer), np.shape(layer)))

    # print("max value", activation_values[0, 48][1968][3][4])
    return layers_mean, layers_max, layers_median


def get_layers_statistics(values):
    imgs_mean = np.zeros(np.shape(values))
    imgs_max = []
    imgs_median = []

    # loop over all images
    for img_idx in tqdm(range(np.shape(values)[0])):
        layers_mean, layers_max, layers_median = get_statistics_per_img(values, img_idx=img_idx)
        imgs_mean[img_idx, :] = layers_mean
        imgs_max.append(layers_max)
        imgs_median.append(layers_median)

    means = [np.mean(imgs_mean, axis=0), np.amin(imgs_mean, axis=0), np.amax(imgs_mean, axis=0)]
    maxes = [np.mean(imgs_max, axis=0), np.amin(imgs_max, axis=0), np.amax(imgs_max, axis=0)]
    medians = [np.mean(imgs_median, axis=0), np.amin(imgs_median, axis=0), np.amax(imgs_median, axis=0)]

    return means, maxes, medians


def get_statistics_in_fn_of_filters(values):
    means = []
    maxes = []

    # loop over filters
    for filt_idx in range(np.shape(values)[0]):
        means.append(np.mean(values[filt_idx]))
        maxes.append(np.amax(values[filt_idx]))

    return means, maxes


def get_filters_statistics(values, n=20, display=False, verbose=False):

    all_max_index = np.zeros((np.shape(values)[1], n))
    all_max_values = np.zeros((np.shape(values)[1], n))
    # loop over every layers
    for layer_idx in range(np.shape(values)[1]):
    # for layer_idx in [17, 39, 48]:
        img_means = []
        img_maxes = []
        # loop over each images
        for img_idx in range(np.shape(values)[0]):
            means, maxes = get_statistics_in_fn_of_filters(values[img_idx, layer_idx])
            img_means.append(means)
            img_maxes.append(maxes)

        means_img_means = np.mean(img_means, axis=0)
        std_means = np.std(img_means, axis=0)
        std_maxes = np.std(img_maxes, axis=0)
        mean_maxes = np.mean(img_maxes, axis=0)

        # get n max activation position
        n_max = np.argsort(mean_maxes)[-n:]
        # switch array position
        n_switch_max = n_max[range(n - 1, -1, -1)]
        all_max_index[layer_idx] = n_switch_max
        all_max_values[layer_idx] = mean_maxes[n_switch_max]

        if verbose:
            # 10 smallest std positions
            # min_std_means = np.argsort(std_means)[:30]
            # print(min_std_means)
            # print(std_means[min_std_means])
            # print("length means", len(std_means[std_means <= 0.]))
            # min_std_max = np.argsort(std_maxes)[:30]
            # print(min_std_max)
            # print(std_means[min_std_max])
            print("length maxes", len(std_maxes[std_maxes <= 0.]))
            print("np.argsort(std_maxes)[:8]", np.argsort(std_maxes)[:8])
            print("np.where(std_maxes == 0)", np.where(std_maxes == 0))
            # print("std_maxes[1469]", std_maxes[1469])
            print("Best 8 activations layers")
            print("np.argsort(mean_maxes)[-8:]", np.argsort(mean_maxes)[-8:])
            print("np.argsort(mean_maxes)[-16:]", np.argsort(mean_maxes)[-16:])

        if display:
            create_plot(1, "Std Layer_%s" % layer_idx)
            add_subplot([2, 1, 1], std_means, np.argsort(std_means)[:8], "Std Means")
            add_subplot([2, 1, 2], std_maxes, np.argsort(std_maxes)[:8], "Std Max")

            create_plot(2, "Std vs. Mean of max, layer_%s" % layer_idx)
            add_subplot([2, 1, 1], std_maxes, np.argsort(std_maxes)[:8], "Std Max")
            add_subplot([2, 1, 2], mean_maxes, np.argsort(mean_maxes)[-8:], "Mean Max")

            plt.show()


    best_n_max_switch = np.unravel_index(np.argsort(all_max_values, axis=None)[-n:], all_max_values.shape)
    best_n_max = np.zeros((3, n))  # (layer_idx, filter_idx)
    for i in range(n):
        best_n_max[0, i] = best_n_max_switch[0][n - 1 - i]
        best_n_max[1, i] = all_max_index[best_n_max_switch[0][n - 1 - i], best_n_max_switch[1][n - 1 - i]]
        best_n_max[2, i] = all_max_values[best_n_max_switch[0][n - 1 - i], best_n_max_switch[1][n - 1 - i]]

    print(np.shape(values)[1])
    plt.hist(best_n_max[0], np.shape(values)[1], [0, np.shape(values)[1]])
    plt.show()

    return best_n_max


def get_statistics_per_layer(values, layer_idx=0):
    images_mean = []
    images_max = []
    images_median = []

    # loop over each images
    for img_idx in range(np.shape(values)[0]):
        layer = activation_values[img_idx, layer_idx]

        images_mean.append(np.mean(layer))
        images_max.append(np.amax(layer))
        images_median.append(np.median(layer))

    return images_mean, images_max, images_median


def create_plot(idx, name):
    plt.figure(idx)
    plt.suptitle(name)


def add_subplot(idx, val1, val2, name):
    x = np.arange(0, len(val1))
    plt.subplot(idx[0], idx[1], idx[2])
    plt.plot(x, val1, '-', val2, val1[val2], 'ro')
    plt.title(name)


def plot_filters(means, maxes):
    fig, ax = plt.subplots()
    x = np.arange(0, len(means))
    ax.plot(x, maxes, 'bo')
    ax.plot(x, means, '-')
    plt.show()


def plot(mean, max_val, median, name=['', '', '']):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    x = np.arange(0, len(mean))
    mean_line, = ax0.plot(x, mean, '-')
    var_line, = ax0.plot(x, median, 'r-')
    ax0.legend([mean_line, var_line], ["Mean", "Variance"])
    ax0.set_title(name[1])
    ax1.plot(x, max_val, 'bo',)
    ax1.set_title(name[2])
    plt.suptitle(name[0])
    plt.show()


def plot_errorbar(mean, max_val, median):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    x = np.arange(0, len(mean[0]))
    ax0.errorbar(x, mean[0], yerr=[mean[0] - mean[1], mean[2] - mean[0]], fmt='-')
    ax0.errorbar(x, median[0], yerr=[median[0] - median[1], median[2] - median[0]], fmt='r-')
    ax1.errorbar(x, max_val[0], yerr=[max_val[0] - max_val[1], max_val[2] - max_val[0]], fmt='bo')
    plt.show()


if __name__ == "__main__":
    activation_values = np.load('../features/activation_values.npy')
    print("np.shape(activation_values)", np.shape(activation_values))

    # *********************************************************************************** #
    # ************************            layer        ********************************** #
    # *********************************************************************************** #
    # layers_mean, layers_max, layers_median = get_statistics_per_img(activation_values, img_idx=0)
    # # print("mean", layers_mean)
    # # print("median", layers_median)
    # # print("max", layers_max)
    # means, maxes, medians = get_layers_statistics(activation_values)
    # print("np.argsort(maxes[0])[-3:]", np.argsort(maxes[0])[-3:])
    #
    # # # get the indexes of the max activations
    # # layer_test = activation_values[0, 38]
    # # print("np.shape", np.shape(layer_test), "argmax", np.amax(layer_test))
    # # print(np.unravel_index(np.argmax(layer_test), np.shape(layer_test)))
    # # print("")
    #
    # plot(layers_mean, layers_max, layers_median, ["Layers Activation", "mean/var", "Max"])
    # plot_errorbar(means, maxes, medians)

    # *********************************************************************************** #
    # ************************            image        ********************************** #
    # *********************************************************************************** #
    # images_mean, images_max, images_median = get_statistics_per_layer(activation_values, layer_idx=48)
    # print("images_max", images_max)
    # plot(images_mean, images_max, images_median, ['Layer 48 in function of images', 'Mean/var', 'Max'])

    # *********************************************************************************** #
    # ************************           filters       ********************************** #
    # *********************************************************************************** #
    n_best = get_filters_statistics(activation_values, n=50)
    print("n_best")
    print(n_best)
