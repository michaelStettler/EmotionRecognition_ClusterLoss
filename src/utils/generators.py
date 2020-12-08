import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import *
from sklearn.datasets import make_blobs
from tensorflow.python.keras.preprocessing.image_dataset import \
    image_dataset_from_directory

sys.path.insert(0, '../')
from src.utils.data_augmentation import *


def get_generator(dataset_parameters,
                  model_parameters,
                  computer_parameters,
                  only_validation: bool = False):

    num_classes = dataset_parameters['num_classes']
    training_data_generator = get_data_generator(dataset_parameters)
    validation_data_generator = get_data_generator(dataset_parameters, False)
    training_generator = None

    if dataset_parameters['dataset_name'] == 'cifar10':
        # load the images from keras
        (training_images, training_labels), (test_images, test_label) \
            = tf.keras.datasets.cifar10.load_data()

        # Normalize data
        training_images = training_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255

        # one-hot-encode the labels
        training_labels = tf.keras.utils. \
            to_categorical(training_labels, num_classes)
        test_label = tf.keras.utils. \
            to_categorical(test_label, num_classes)

        # create generators for easier use in model.fit()
        training_generator = training_data_generator.flow(
            training_images,
            training_labels,
            batch_size=model_parameters['batch_size']
        )
        validation_generator = validation_data_generator.flow(
            test_images,
            test_label
        )

    elif 'csv' in dataset_parameters['labels_type']:
        if not only_validation:
            training_csv_file = os.path.join(computer_parameters['dataset_path'],
                                             dataset_parameters['csv_training_file'])
            training_directory = os.path.join(computer_parameters['dataset_path'],
                                              dataset_parameters['training_directory'])

            training_dataframe = pd.read_csv(training_csv_file)

            training_generator = training_data_generator.flow_from_dataframe(
                dataframe=training_dataframe,
                directory=training_directory,
                x_col='subDirectory_filePath',
                y_col=dataset_parameters['class_label'],
                class_mode='categorical',
                target_size=(model_parameters['image_height'],
                             model_parameters['image_width']),
                batch_size=model_parameters['batch_size'],
                shuffle=True
            )

        validation_csv_file = os.path.join(computer_parameters['dataset_path'],
                                           dataset_parameters['csv_validation_file'])
        validation_directory = os.path.join(computer_parameters['dataset_path'],
                                            dataset_parameters['validation_directory'])

        validation_dataframe = pd.read_csv(validation_csv_file)

        validation_generator = validation_data_generator.flow_from_dataframe(
            dataframe=validation_dataframe,
            directory=validation_directory,
            x_col='subDirectory_filePath',
            y_col=dataset_parameters['class_label'],
            class_mode='categorical',
            target_size=(model_parameters['image_height'],
                         model_parameters['image_width']),
            batch_size=model_parameters['batch_size'],
            shuffle=False,
        )

    elif 'directory' in dataset_parameters['labels_type']:
        training_directory = dataset_parameters['training_directory']
        validation_directory = dataset_parameters['validation_directory']

        if not os.path.isdir(training_directory):
            raise ValueError('Training directory does not exist',
                             training_directory)
        if not os.path.isdir(validation_directory):
            raise ValueError('Validation directory does not exist',
                             validation_directory)

        training_generator = image_dataset_from_directory(
            training_directory,
            validation_split=0.2,
            subset="training",
            label_mode='categorical',
            class_names=dataset_parameters['class_names'],
            batch_size=model_parameters['batch_size'],
            image_size=(model_parameters['img_height'],
                        model_parameters['image_width']),
            shuffle=True,
        )
        validation_generator = image_dataset_from_directory(
            validation_directory,
            validation_split=0.2,
            subset="validation",
            label_mode='categorical',
            class_names=dataset_parameters['class_names'],
            image_size=(model_parameters['image_height'],
                        model_parameters['image_width'])
        )

    # creates a simple blob dataset, which is fast to train
    elif dataset_parameters['dataset_name'] == 'blob':
        features = model_parameters['image_height'] \
                   * model_parameters['image_width'] * 3

        training_data, training_labels = make_blobs(n_samples=1000,
                                                    n_features=features,
                                                    centers=num_classes)

        training_data = np.reshape(training_data,
                                   (1000,
                                    model_parameters['image_height'],
                                    model_parameters['image_width'],
                                    3))

        _, test_data = np.split(training_data, [800])
        _, test_label = np.split(training_labels, [800])

        training_labels = tf.keras.utils.to_categorical(training_labels,
                                                        num_classes)
        test_label = tf.keras.utils.to_categorical(test_label,
                                                   num_classes)

        training_generator = training_data_generator.flow(
            training_data,
            training_labels,
            batch_size=model_parameters['batch_size']
        )
        validation_generator = validation_data_generator.flow(
            test_data,
            test_label
        )

    return training_generator, validation_generator


def get_cluster_generator(dataset_parameters,
                          model_parameters,
                          computer_parameters,
                          only_validation: bool = False):

    # get generator
    train_gen, val_gen = get_generator(dataset_parameters,
                                       model_parameters,
                                       computer_parameters,
                                       only_validation)

    def cluster_generator(generator, batch_size, num_entry):
        iter = 0

        while True:
            batch = next(generator)
            img = batch[0]
            labels = batch[1]
            cluster = np.zeros(batch_size)

            x = [img, labels]
            y = [labels, cluster]
            yield x, y

            # iter += batch_size
            # if iter >= num_entry:
            #     break

    training_generator = cluster_generator(train_gen, model_parameters['batch_size'], 287650)
    validation_generator = cluster_generator(val_gen, model_parameters['batch_size'], 4000)

    return training_generator, validation_generator

    # https://github.com/keras-team/keras/issues/8130

    # https://github.com/keras-team/keras/issues/3386
    # while True:
    #     # suffled indices
    #     idx = np.random.permutation(X.shape[0])
    #     # create image generator
    #     datagen = ImageDataGenerator(
    #         featurewise_center=False,  # set input mean to 0 over the dataset
    #         samplewise_center=False,  # set each sample mean to 0
    #         featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #         samplewise_std_normalization=False,  # divide each input by its std
    #         zca_whitening=False,  # apply ZCA whitening
    #         rotation_range=10,  # 180,  # randomly rotate images in the range (degrees, 0 to 180)
    #         width_shift_range=0.1,  # 0.1,  # randomly shift images horizontally (fraction of total width)
    #         height_shift_range=0.1,  # 0.1,  # randomly shift images vertically (fraction of total height)
    #         horizontal_flip=False,  # randomly flip images
    #         vertical_flip=False)  # randomly flip images
    #
    #     batches = datagen.flow(X[idx], Y[idx], batch_size=64, shuffle=False)
    #     idx0 = 0
    #     for batch in batches:
    #         idx1 = idx0 + batch[0].shape[0]
    #
    #         yield [batch[0], I[idx[idx0:idx1]]], batch[1]
    #
    #         idx0 = idx1
    #         if idx1 >= X.shape[0]:
    #             break
