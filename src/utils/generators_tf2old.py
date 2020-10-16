import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import *
from sklearn.datasets import make_blobs

sys.path.insert(0, './')
from data_augmentation import *


def get_generator(dataset_parameters,
                  model_parameters,
                  only_validation: bool = False):

    num_classes = dataset_parameters['num_classes']
    training_data_generator = get_data_generator(dataset_parameters)
    validation_data_generator = get_data_generator(dataset_parameters, False)

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
            training_dataframe = pd.read_csv(
                model_parameters['csv_training_file'])

            training_generator = training_data_generator.flow_from_dataframe(
                dataframe=training_dataframe,
                directory=dataset_parameters['training_directory'],
                x_col='subDirectory_filePath',
                y_col=dataset_parameters['class_label'],
                has_ext=True,
                class_mode='categorical',
                target_size=(model_parameters['image_height'],
                             model_parameters['image_width']),
                batch_size=model_parameters['batch_size'],
                shuffle=True
            )

        validation_dataframe = pd.read_csv(
            model_parameters['csv_validation_file'])

        validation_generator = validation_data_generator.flow_from_dataframe(
            dataframe=validation_dataframe,
            directory=dataset_parameters['validation_directory'],
            x_col='subDirectory_filePath',
            y_col=dataset_parameters['class_label'],
            has_ext=True,
            class_mode='categorical',
            target_size=(model_parameters['image_height'],
                         model_parameters['image_width']),
            batch_size=model_parameters['batch_size'],
            shuffle=False,
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

        training_data, test_data = np.split(training_data, [800])
        training_labels, test_label = np.split(training_labels, [800])

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

