from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import shutil
import sys
import numpy as np
import os
import datetime
import cv2

from six.moves import cPickle
import keras
from keras import backend as K
from keras import layers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = '../../../data/raw/Cifar-10/'
BATCH_SIZE = 128


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads CIFAR10 dataset.
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
    file_path = '../../../data/raw/Cifar-10/cifar-10-python.tar'
    dir_path = '../../../data/raw/Cifar-10/'

    open_fn = tarfile.open
    is_match_fn = tarfile.is_tarfile

    if is_match_fn(file_path):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(dir_path)
                print("Successfully extracted all files")
            except (tarfile.TarError, RuntimeError,
                    KeyboardInterrupt):
                if os.path.exists(dir_path):
                    if os.path.isfile(dir_path):
                        os.remove(dir_path)
                    else:
                        shutil.rmtree(dir_path)
                raise

    else:
        print("Problem with data")

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dir_path+'cifar-10-batches-py', 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(dir_path+'cifar-10-batches-py', 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # set up for tensorflow backend using 'channels_last'
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    # 'channels_last':
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    # 'channels_last':
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_cifar(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters

    # 'channels_last':
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, kernel_size, name=conv_name_base + '2a', strides=strides, padding='same',
               kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, strides=(1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_cifar(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters

    # 'channels_last':
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, kernel_size, name=conv_name_base + '2a', strides=strides, padding='same',
               kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(filters2, (1, 1), strides=strides,
                      name=conv_name_base + '1', kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet(num_layers=32):

    n = (num_layers - 2) // 6
    print("n", n)
    # for 'channel_last'
    bn_axis = 3

    img_input = Input(shape=(32, 32, 3))

    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(img_input)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='valid', name='conv1', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    # x = ZeroPadding2D(padding=(1, 1), name='pad_1')(x)

    for i in range(2 * n):
        x = identity_block_cifar(x, 3, [16, 16], stage=2, block=str(i), strides=(1, 1))

    for i in range(2 * n):
        if i == 0:
            x = conv_block_cifar(x, 3, [32, 32], stage=3, block=str(i), strides=(2, 2))
        else:
            x = identity_block_cifar(x, 3, [32, 32], stage=3, block=str(i), strides=(1, 1))

    for i in range(2 * n):
        if i == 0:
            x = conv_block_cifar(x, 3, [64, 64], stage=4, block=str(i), strides=(2, 2))
        else:
            x = identity_block_cifar(x, 3, [64, 64], stage=4, block=str(i), strides=(1, 1))

    # x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((8, 8), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(10, activation='softmax', name='fc1000')(x)

    return Model(inputs=img_input, outputs=x, name='resNet'+str(num_layers))

if __name__ == '__main__':
    print("Own implementation of ResNet 32 using cifar-10 dataset")
    start = datetime.datetime.now()
    print("started at:", start)
    # load the data into numpy array
    (x_train, y_train), (x_test, y_test) = load_data()
    print("loaded cifar 10 dataset")

    # one hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # print("x_train shape", np.shape(x_train))
    # print("y_train shape", np.shape(y_train))
    # print("x_test shape", np.shape(x_test))
    # print("y_test shape", np.shape(y_test))

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # initially True
        featurewise_std_normalization=False,  # initially True
        rotation_range=0,  # initially 20
        width_shift_range=0,  # initially 0.2
        height_shift_range=0,  # initially 0.2
        horizontal_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)


    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]


    def crop_generator(batches, crop_length):
        '''
        Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator
        '''
        crop_batch = np.zeros((batches.shape[0], crop_length, crop_length, 3))
        # while True:
        #     x = x + 1
        #     print(x)
        #     batch_x, batch_y = next(batches)
        #     batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        #     for i in range(batch_x.shape[0]):
        #         batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        #     yield (batch_crops, batch_y)
        for i, img in enumerate(batches):
            crop_img = random_crop(img, (crop_length, crop_length))
            crop_batch[i] = crop_img
        return crop_batch


    x_train_padded = np.zeros((np.shape(x_train)[0], 40, 40, 3))
    for i, img in enumerate(x_train):
        image = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_REPLICATE)  #BORDER_CONSTANT cv2.BORDER_REPLICATE cv2.BORDER_REFLECT cv2.BORDER_DEFAULT
        x_train_padded[i] = image

    print("np.shape(x_train_padded)", np.shape(x_train_padded))
    x_train = crop_generator(x_train_padded, 32)
    print("np.shape(x_train)", np.shape(x_train))

    # build ResNet 32 model
    model = resnet(num_layers=32)

    # print(model.summary())

    # compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['mae', 'accuracy'])
    print(model.summary())

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=len(x_train) / 128, epochs=80)

    K.set_value(model.optimizer.lr, .01)
    print("new learning rate of 0.01 for next 40 epochs")
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=40)

    K.set_value(model.optimizer.lr, .001)
    print("new learning rate of 0.001 for next 40 epochs")
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=40)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("loss", loss_and_metrics[0], "mae", loss_and_metrics[1], "accuracy", loss_and_metrics[2])

    # classes = model.predict(x_test, batch_size=128)
    # print(classes)

    print("total running time:", datetime.datetime.now() - start)

