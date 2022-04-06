from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.applications import *


def resnet_generator(train: bool):
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet.preprocess_input,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=30)
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    return data_generator


def resnet_v2_generator(train: bool):
    print("** ResNetv2 generator")
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=30)
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    return data_generator


def resnet_v2_none_generator(train: bool):
    print("** ResNetv2 None_generator")
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    return data_generator


def vgg_generator(train: bool):
    print("** VGG generator")
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=vgg16.preprocess_input,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=30)
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=vgg16.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    return data_generator

def vgg_none_generator(train: bool):
    print("** VGG None generator")
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=vgg16.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=vgg16.preprocess_input,
            horizontal_flip=False,
            fill_mode="nearest")
    return data_generator



def resnet_basic_generator(train: bool):
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet.preprocess_input)
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet.preprocess_input)
    return data_generator


def resnet_v2_basic_generator(train: bool):
    if train:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input)
    else:
        data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=resnet_v2.preprocess_input)
    return data_generator


def get_data_generator(dataset_parameters, train=True):
    def preprocess_fn(x):
        mean_rgb = dataset_parameters['data_augmentation']['mean_rgb']
        std_rgb = dataset_parameters['data_augmentation']['std_rgb']

        x[:, :, 0] = (x[:, :, 0] - mean_rgb[0]) / std_rgb[0]
        x[:, :, 1] = (x[:, :, 1] - mean_rgb[1]) / std_rgb[1]
        x[:, :, 2] = (x[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return x

    # No Data augmentation
    if not dataset_parameters['data_augmentation']:
        data_generator = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=False)

    elif dataset_parameters['data_augmentation'] == 'resnet_basic':
        data_generator = resnet_basic_generator(train)

    elif dataset_parameters['data_augmentation'] == 'resnetv2_basic':
        data_generator = resnet_v2_basic_generator(train)

    elif dataset_parameters['data_augmentation'] == 'resnet_augmentation':
        data_generator = resnet_generator(train)

    elif dataset_parameters['data_augmentation'] == 'resnetv2_augmentation':
        data_generator = resnet_v2_generator(train)
    elif dataset_parameters['data_augmentation'] == 'resnetv2_none_augmentation':
        data_generator = resnet_v2_none_generator(train)
    elif dataset_parameters['data_augmentation'] == 'vgg_augmentation':
        data_generator = vgg_generator(train)
    elif dataset_parameters['data_augmentation'] == 'vgg_none_augmentation':
        data_generator = vgg_none_generator(train)

    # random horizontal flips, rotation and size changes
    elif dataset_parameters['data_augmentation'] == 1:
        if train:
            data_generator = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                fill_mode="nearest",
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=30)
        else:
            data_generator = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=False,
                fill_mode="nearest",
                zoom_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                rotation_range=0.)

    elif dataset_parameters['data_augmentation'] == 2:

        if train:
            data_generator = ImageDataGenerator(
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
                rescale=1. / 255,
                # set function that will be applied on each input
                preprocessing_function=preprocess_fn,
                # fraction of images reserved for validation
                # (strictly between 0 and 1)
                validation_split=0.0)
        else:
            data_generator = ImageDataGenerator(
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
                rescale=1. / 255,
                # set function that will be applied on each input
                preprocessing_function=preprocess_fn,
                # fraction of images reserved for validation
                # (strictly between 0 and 1)
                validation_split=0.0)

    else:
        raise ValueError('Wrong data_processing augmentation selection - '
                         '%s doest not exist'
                         % dataset_parameters['data_augmentation'])

    return data_generator
