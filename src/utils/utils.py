import numpy as np
import cv2
from scipy.misc import imsave
import os
from os import listdir
from os.path import isfile, join
import pickle
from keras.utils import np_utils
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_metrics(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
    plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    plt.show()


def stiched_and_save_filter_to_img(images, num_column, num_row, img_width, img_height, layer_printed_name="name", marrgin=5):
    # build a black picture with enough space for
    # our 8 x num_column filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = num_column * img_width + (num_column - 1) * margin
    height = num_row * img_height + (num_row - 1) * margin
    stitched_filters = np.zeros((height, width, 3))

    # fill the picture with our saved filters
    for i in range(num_row):
        for j in range(num_column):
            try:
                img, loss = images[i * num_column + j]
            except IndexError:
                img = np.zeros((img_height, img_width, 3))
            except ValueError:
                img = images[i * num_column + j]

            stitched_filters[(img_height + margin) * i: (img_height + margin) * i + img_height,
                             (img_width + margin) * j: (img_width + margin) * j + img_width, :] = img

    # save the result to disk
    # image.save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    imsave('%s_filter_%d.png' % (layer_printed_name, len(images)), stitched_filters)


def save_img(img, name):
    imsave('%s.png' % name, img)


def import_img_name_from_files(path):
    images = []

    #get all files within the folder
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        # add the path to the name
        images.append(os.path.join(path, file))

    return images


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def process_cifar10(x_train, y_train, x_test, y_test, num_classes, img_width, img_height):
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    new_x_train = np.zeros((np.shape(x_train)[0], img_height, img_width, 3))
    new_x_test = np.zeros((np.shape(x_test)[0], img_height, img_width, 3))
    for i, img in enumerate(tqdm(x_train)):
        # expand the image
        new_x_train[i] = cv2.resize(img, dsize=(img_height, img_width),
                         interpolation=cv2.INTER_AREA)

        # add border to the image
        # new_img = np.zeros((img_width, img_height, 3))
        # new_img[96:128, 96:128, :] = img
        # new_x_train[i] = new_img

    for i, img in enumerate(tqdm(x_test)):
        new_x_test[i] = cv2.resize(img, dsize=(img_height, img_width),
                         interpolation=cv2.INTER_AREA)

    return new_x_train, y_train, new_x_test, y_test


def get_args_from_weights(weights):
    split = weights.split('_')
    run = split[-1]
    run = run.split('.')[0]
    if 'cw-' in weights:
        cw = split[-2]
        cw = cw.split('-')[-1]
        version = split[-3]
        version = version.split('-')[-1]
        da = split[-4]
        da = da.split('-')[-1]
    else:
        version = split[-2]
        version = version.split('-')[-1]
        da = split[-3]
        da = da.split('-')[-1]
        cw = None
    model = split[0]
    task = split[1]
    dataset = split[2]

    if model == 'checkpoint':
        model = split[1]
        task = split[2]
        dataset = split[3]

    return model, dataset, run, version, task, da, cw


if __name__ == "__main__":
    # images = import_img_name_from_files("../../data_processing/processed/Maya/Face/rotate_hori")
    # print(images)
    # test_dict = {'test': 3, 'dict2': {'tes2': 4, 'test8': 5}}
    # save_obj(test_dict, "YOLO")
    obj = load_obj("YOLO")
    print(obj)
