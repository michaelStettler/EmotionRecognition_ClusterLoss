import sys
import os
import numpy as np
import pandas as pd

import keras
from keras import applications
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

sys.path.insert(0, '../../models/ResNet/keras/')
sys.path.insert(0, '../../models/Simple')
from resnet18 import *
from resnet50 import *
from resnet50v2 import *
from keras.applications.densenet import DenseNet201
from simple import *


def get_generator(data, model_params, data_augmentation, task, validation_only=False):
    if data['labels_type'] == 'numpy':
        if data['dataset'] == 'cifar10':
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train, y_train, x_test, y_test = process_cifar10(x_train, y_train, x_test, y_test,
                                                               data['n_classes'],
                                                               model_params['img_width'],
                                                               model_params['img_height'])
        else:
            print("problem with the dataset")
        # use the flow method
        generators = get_data_generator(x_train, y_train, x_test, y_test, data, model_params, data_augmentation, validation_only)
    elif data['labels_type'] == 'auto':
        # use the flow_from_directory method and thus the labels are automatically generated
        generators = get_auto_generator(data, model_params, data_augmentation, validation_only)
    elif data['labels_type'] == 'csv':
        # use the flow_from_dataframe method and thus the labels are read from the file
        generators = get_csv_generator(data, model_params, data_augmentation, validation_only, task=task)

    if validation_only:
        return generators[0]
    else:
        return generators[0], generators[1]


def get_data_generator(x_train, y_train, x_val, y_val, data, model_params, data_augmentation, validation_only):
    # get the data generator
    if not validation_only:
        train_datagen = get_datagen(data_augmentation, data, train=True)
    val_datagen = get_datagen(data_augmentation, data, train=False)

    if not validation_only:
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=model_params['batch_size'],
            #save_to_dir='test_train'
        )

    validation_generator = val_datagen.flow(
        x_val, y_val,
        shuffle=False,
        # save_to_dir='test_val'
    )

    if validation_only:
        return [validation_generator]
    else:
        return [train_generator, validation_generator]


def get_auto_generator(data, model_params, data_augmentation, validation_only):
    if not validation_only:
        train_data_dir = data['train_dir']
    validation_data_dir = data['val_dir']

    if not validation_only:
        if not os.path.isdir(train_data_dir):
            raise ValueError('Train directory does not exist', train_data_dir)
    if not os.path.isdir(validation_data_dir):
        raise ValueError('Validation directory does not exist', validation_data_dir)

    # get the data generator
    if not validation_only:
        train_datagen = get_datagen(data_augmentation, data, train=True)
    val_datagen = get_datagen(data_augmentation, data, train=False)

    if not validation_only:
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(model_params['img_height'], model_params['img_width']),
            batch_size=model_params['batch_size'],
            class_mode="categorical",
            #save_to_dir='test_train'
        )

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(model_params['img_height'], model_params['img_width']),
        class_mode="categorical",
        shuffle=False,
        # save_to_dir='test_val'
    )

    if validation_only:
        return [validation_generator]
    else:
        return [train_generator, validation_generator]


def get_csv_generator(data, model_params, data_augmentation, validation_only, task):
    # get the data generator
    if not validation_only:
        train_datagen = get_datagen(data_augmentation, data, train=True)
    val_datagen = get_datagen(data_augmentation, data, train=False)

    if not validation_only:
        df_train = pd.read_csv(data['csv_train_file'])
    df_val = pd.read_csv(data['csv_val_file'])

    # affect net comes with the subfolder path, we need to remove it in order for flow_from_dataframe to work
    # Use this if we want to use the given training.csv file
    # if data['dataset'] == 'affectnet':
        # for index, row in df_train.iterrows():
        # df_train.at[index, 'subDirectory_filePath'] = row['subDirectory_filePath'].split('/')[1]
        # for index, row in df_val.iterrows():
        #     df_val.at[index, 'subDirectory_filePath'] = row['subDirectory_filePath'].split('/')[1]

    if task == 'classification':
        class_mode = 'categorical'
        y_col = data['class_label']
    elif task == 'regression':
        class_mode = 'other'
        y_col = data['box_labels']
    else:
        raise ValueError('Wrong task selected - %s doest not exist' % task)

    if not validation_only:
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            directory=data['train_dir'],
            x_col='subDirectory_filePath',
            y_col=y_col,
            has_ext=True,
            class_mode=class_mode,
            target_size=(model_params['img_height'], model_params['img_width']),
            batch_size=model_params['batch_size'],
            shuffle=True,
            # save_to_dir='test_train'
        )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=data['val_dir'],
        x_col='subDirectory_filePath',
        y_col=y_col,
        has_ext=True,
        class_mode=class_mode,
        target_size=(model_params['img_height'], model_params['img_width']),
        batch_size=model_params['batch_size'],
        shuffle=False,
        #save_to_dir='test_val'
    )

    # print("fit data")
    # train_datagen.fit(train_generator)
    # val_datagen.fit(validation_generator)

    if validation_only:
        return [validation_generator]
    else:
        return [train_generator, validation_generator]


def get_datagen(data_augmentation, data, train=True):
    if data_augmentation == '0':  # No Data augmentation
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=False)

    elif data_augmentation == '1':
        if train:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                fill_mode="nearest",
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=30)
        else:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=False,
                fill_mode="nearest",
                zoom_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                rotation_range=0.)
    elif data_augmentation == '2':

        def preprocess_fn(x):
            x[:, :, 0] = (x[:, :, 0] - data['mean_RGB'][0])/data['std_RGB'][0]
            x[:, :, 1] = (x[:, :, 1] - data['mean_RGB'][1])/data['std_RGB'][1]
            x[:, :, 2] = (x[:, :, 2] - data['mean_RGB'][2])/data['std_RGB'][2]
            return x

        if train:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=30,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.3,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=1./255,
                # set function that will be applied on each input
                preprocessing_function=preprocess_fn,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
        else:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.,
                # randomly shift images vertically
                height_shift_range=0.,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=False,
                # randomly flip images *
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=1./255,
                # set function that will be applied on each input
                preprocessing_function=preprocess_fn,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

    else:
        raise ValueError('Wrong data augmentation selection - %s doest not exist' % data_augmentation)

    return datagen


def load_loss(loss_type, model_params):
    loss = None
    if loss_type == 'categorical_crossentropy':
        loss = keras.losses.categorical_crossentropy
    elif loss_type == 'constrained_categorical_loss':
        # constrain the loss just to ensure that they are not going to be 0 and create infinite number
        def constrainedCrossEntropy(ytrue, ypred):
            ypred = keras.backend.clip(ypred, 0.0001, 0.9999)
            return keras.losses.categorical_crossentropy(ytrue, ypred)

        loss = constrainedCrossEntropy
    elif loss_type == 'weighted_categorical_loss':
        def weighted_categorical_loss(weights):
            weights = K.variable(weights)

            def loss(y_true, y_pred):
                # scale predictions so that the class probas of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
                # clip to prevent NaN's and Inf's
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                # calc
                loss = y_true * K.log(y_pred) * weights
                loss = -K.sum(loss, -1)
                return loss

            return loss

        if model_params['class_weights'] is None:
            sys.exit('Weighted loss is selected but no weights were given!')
        else:
            print("class weights:", model_params['class_weights'])

        weights = model_params['class_weights']
        loss = weighted_categorical_loss(weights)
    else:
        print("Please select a valid loss: (%s) Not Valid" % loss_type)

    return loss


def load_model(include_top, weights, model_params, data):
    # load the model
    if 'resnet18' in model_params['name']:
        model_template = ResNet18(include_top=include_top,
                                  weights=weights,
                                  classes=data['n_classes'],
                                  input_shape=(model_params['img_width'], model_params['img_height'], 3),
                                  l2_reg=model_params['l2_reg'])

    elif 'resnet50v2' in model_params['name']:
        model_template = ResNet50v2(include_top=include_top,
                                  weights=weights,
                                  classes=data['n_classes'],
                                  input_shape=(model_params['img_width'], model_params['img_height'], 3),
                                  l2_reg=model_params['l2_reg'])
        print("Loaded resnet v2")

    elif 'resnet50Keras' in model_params['name']:
        print("Load resnet50 from keras")
        model_template = applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                               input_shape=(model_params['img_width'], model_params['img_height'], 3),
                                               pooling=None)
    elif 'resnet50' in model_params['name']:
        model_template = ResNet50(include_top=include_top,
                                  weights=weights,
                                  classes=data['n_classes'],
                                  input_shape=(model_params['img_width'], model_params['img_height'], 3),
                                  l2_reg=model_params['l2_reg'])
        print("Loaded resnet v1")

    elif 'densenet201' in model_params['name']:
        model_template = DenseNet201(include_top=include_top,
                         weights=weights,
                         classes=data['n_classes'],
                         input_shape=(model_params['img_width'], model_params['img_height'], 3))

    elif 'vgg16' in model_params['name']:
        model_template = VGG16(include_top=include_top,
                      weights=weights,
                      classes=data['n_classes'],
                      input_shape=(model_params['img_width'], model_params['img_height'], 3))

    elif 'simple' in model_params['name']:
        model_template = SIMPLE(include_top=include_top,
                      weights=weights,
                      classes=data['n_classes'],
                      input_shape=(model_params['img_width'],  model_params['img_height'], 3),
                      l2_reg=model_params['l2_reg'])
    else:
        raise ValueError('Please select a model')

    # construct multi GPU if possible
    if len(device_lib.list_local_devices()) > 2:
        model = multi_gpu_model(model_template)
    else:
        model = model_template

    # return the model template for saving issues with multi GPU
    return model, model_template