from __future__ import division
import numpy as np
import math

np.set_printoptions(precision=5, linewidth=400, suppress=True)

filt1 = np.array([[0.3, 0.5, 0.4],
                  [0.7, 0.8, 0.1],
                  [0.2, 0.9, 0.6]])

filt2 = np.array([[0.5, 0.7, 0.2],
                  [0.1, 0.9, 0.4],
                  [0.6, 0.8, 0.3]])

filt8 = np.array([[0.2, 0.5, 0.1],
                  [0.5, 0.7, 0.8],
                  [0.3, 0.2, 0.6]])

filt9 = np.array([[0.1, 0.8, 0.3],
                  [0.3, 0.2, 0.5],
                  [0.2, 0.5, 0.3]])

filt3 = np.array([[0.1, 0.6],
                  [0.2, 0.8]])

filt4 = np.array([[0.3, 0.9],
                  [0.5, 0.7]])

filt5 = np.array([[0.7, 0.3],
                  [0.5, 0.2]])

filt6 = np.array([[0.3, 0.5, 0.4, 0.6, 0.3],
                  [0.7, 0.8, 0.1, 0.4, 0.1],
                  [0.5, 0.6, 0.3, 0.6, 0.5],
                  [0.3, 0.1, 0.8, 0.5, 0.4],
                  [0.2, 0.9, 0.6, 0.9, 0.2]])

filt7 = np.array([[0.1, 0.2, 0.5, 0.3, 0.9],
                  [0.5, 0.5, 0.6, 0.9, 0.2],
                  [0.2, 0.8, 0.5, 0.3, 0.4],
                  [0.8, 0.2, 0.6, 0.5, 0.2],
                  [0.4, 0.5, 0.7, 0.4, 0.1]])

def deconv_5x5_to_3x3(filt1, filt2):
    filt = np.zeros((5, 5))

    mid_i = int(filt.shape[0]/2)+1
    mid_j = int(filt.shape[1] / 2) + 1

    # loop over all the new entries for the new filter
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            # compute the number of operation to do for each weights of the new filter
            # num_op = (mid - |mid - i|)(mid - |mid - j|)
            max_k = (mid_i - abs(mid_i - i - 1))
            max_l = (mid_j - abs(mid_j - j - 1))

            w = 0
            shift_k = max(0, i - filt1.shape[0] + 1)
            shift_l = max(0, j - filt1.shape[1] + 1)
            for k in range(max_k):
                for l in range(max_l):
                    w += filt1[shift_k + k, shift_l + max_l - l - 1] * filt2[shift_k + max_k - k - 1, shift_l + l]

            filt[i, j] = w

    return filt


def deconv_array(filters):
    # NOT WORKING!!!!!!!

    # compute final filter size
    m, n = 0, 0
    output_size_m, output_size_n = 1, 1
    for filter in filters:
        output_size_m = output_size_m + np.shape(filter)[0] - 1
        output_size_n = output_size_n + np.shape(filter)[1] - 1
        m = output_size_m
        n = output_size_n

    # initialize final filter with zeros
    filt = np.zeros((m, n))

    mid_i = int(filt.shape[0] / 2) + 1
    mid_j = int(filt.shape[1] / 2) + 1
    print("mid_i", mid_i, "mid_j", mid_j)

    # loop over all the new entries for the new filter
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            print("_________________")
            print("i", i, "j", j)
            filt_1 = np.zeros((3, 3))
            filt_2 = np.zeros((3, 3))

            max_k = (mid_i - abs(mid_i - i - 1))
            max_l = (mid_j - abs(mid_j - j - 1))

            shift_k = max(0, i - filters[0].shape[0] + 1)
            shift_l = max(0, j - filters[0].shape[1] + 1)

            filt_1[:max_k, :max_l] = filters[0][:max_k, :max_l]
            print("filt_1")
            print(filt_1)
            filt_2[:max_k, :max_l] = np.rot90(np.rot90(filters[1]))[-max_k:, -max_l:]
            print("filt_2")
            print(filt_2)
            final = np.sum(filt_1 * filt_2)
            print("final")
            print(final)
            filt[i, j] = final

    return filt


def deconv_2(filters, verbose=False):
    # compute final filter size
    m, n = 0, 0
    output_size_m, output_size_n = 1, 1
    for filter in filters:
        output_size_m = output_size_m + np.shape(filter)[0] - 1
        output_size_n = output_size_n + np.shape(filter)[1] - 1
        m = output_size_m
        n = output_size_n
    # initialize final filter with zeros
    filt = np.zeros((m, n))

    mid_i = int(filt.shape[0] / 2) + 1
    # mid_i = int(filt.shape[0] / 2)
    mid_j = int(filt.shape[1] / 2) + 1
    # mid_j = int(filt.shape[1] / 2)

    if verbose:
        print("m", m, "n", n)
        # print("mid_i", mid_i, "mid_j", mid_j)

    # loop over all the new entries for the new filter
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            # compute the number of operation to do for each weights of the new filter
            # num_op = (mid - |mid - i|)(mid - |mid - j|)
            # max_k = mid_i - abs(mid_i - i - 1)
            max_k = min(i+1, filters[1].shape[0])
            # # max_l = mid_j - abs(mid_j - j - 1)
            max_l = min(j+1, filters[1].shape[1])

            w = 0
            shift0_k = max(0, i - filters[1].shape[0] + 1)
            shift0_l = max(0, j - filters[1].shape[1] + 1)

            if verbose:
                print("----------------")
                print("i,j", i, j)
                # print("max_k", max_k, "max_l", max_l)
                print("shift0_k", shift0_k, "shift0_l", shift0_l)

            for k in range(max_k):
                for l in range(max_l):
                    idx0 = [shift0_k + max_k - k - 1, shift0_l + max_l - l - 1]
                    idx1 = [k, l]
                    # if verbose:
                    #     print(idx0, idx1)
                    if idx0[0] < np.shape(filters[0])[0] and idx0[1] < np.shape(filters[0])[1] \
                            and idx1[0] < np.shape(filters[1])[0] and idx1[1] < np.shape(filters[1])[1]:
                        if verbose:
                            print("w0", idx0, "w1", idx1)
                            print()
                        w += filters[0][idx0[0], idx0[1]] * filters[1][idx1[0], idx1[1]]

            filt[i, j] = w

    return filt


# filt = deconv_5x5_to_3x3(filt1, filt2)
# filt = deconv([filt1, filt2])
# filt = deconv_2([filt6, filt7])
# filt = deconv_2([filt6, filt1])

# 3x3 - 3x3
# filt = deconv_2([filt1, filt2], True)
# print(filt)

# 5x5 - 5x5
# filt = deconv_2([filt6, filt7])
# print(filt)

# 3x3 - 3x3 - 3x3
filt = deconv_2([filt1, filt2])
# print(filt)
filt_1 = deconv_2([filt, filt8])
print(filt_1)

# # 3x3 - 3x3 - 3x3 - 3x3
# filt = deconv_2([filt1, filt2])
# # print(filt)
# filt_1 = deconv_2([filt8, filt9])
# # print(filt_1)
# filt_2 = deconv_2([filt, filt_1])
# print(filt_2)

# 5x5 - 3x3
# filt = deconv_2([filt1, filt2])
# filt_2 = deconv_2([filt, filt8], True)
# # print(filt)
# print(filt_2)

# filt = deconv_array([filt3, filt4])
# print(filt)

# if __name__ == '__main__':
#     import keras
#     from keras.models import Sequential
#     from keras.layers import Dense
#     from keras.layers import Input
#     from keras import backend as K
#     from keras.layers import Conv2D
#     from keras.layers import Deconvolution2D
#     from keras.layers import Reshape
#     from keras.models import Model
#
#     training = np.zeros((3, 5, 5, 1))
#     im_1 = np.array([[1.,2.,3.,4.,5.],
#                       [6.,7.,8.,9.,10],
#                       [11,12,13,14,15],
#                       [16,17,18,19,20],
#                       [21,22,23,24,25]])
#     im_1 = np.reshape(im_1, (5, 5, 1))
#     training[1] = im_1
#
#     # constru t a small model with two layers
#     model = Sequential()
#     model.add(Conv2D(1, (3, 3), input_shape=(5, 5, 1)))
#     model.add(Conv2D(1, (3, 3)))
#     model.add(Deconvolution2D(1, 5, 5, output_shape=(5, 5, 1)))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='sgd',
#                   metrics=['accuracy'])
#
#     model.summary()
#
#     # set up the filters weights
#     layer_dict = dict([(layer.name, layer) for layer in model.layers[:-1]])
#     filters = [[[0.3, 0.5, 0.4], [0.7, 0.8, 0.1], [0.2, 0.9, 0.6]],
#                [[0.5, 0.7, 0.2], [0.1, 0.9, 0.4], [0.6, 0.8, 0.3]]]
#     for i, layer in enumerate(layer_dict):
#         print("layer", layer)
#         new_weights = layer_dict[layer].get_weights()
#         new_weights[0][:, :, 0, 0] = filters[i]
#         layer_dict[layer].set_weights(new_weights)
#
#     # compute the convolution results
#     results = []
#     X = training[1:2]
#     inputs = [K.learning_phase()] + model.inputs
#     for layer in layer_dict:
#         _convout_f = K.function(inputs, [layer_dict[layer].output])
#
#         def convout_f(X):
#             # The [0] is to disable the training phase flag
#             return _convout_f([0] + [X])
#
#         x = convout_f(X)
#         results.append(x)
#
#     print(results)
#
#     # deconvolution
#     # todo set the filter weights
