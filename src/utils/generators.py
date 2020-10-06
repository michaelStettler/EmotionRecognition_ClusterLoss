import tensorflow as tf
from tensorflow.keras.preprocessing.image import *
from sklearn.datasets import make_blobs


def get_generator(dataset_parameters, model_parameters):
    num_classes = dataset_parameters['num_classes']

    if dataset_parameters['dataset_name'] == 'cifar10':
        # load the images from keras
        (training_images, training_labels), (test_images, test_label) \
            = tf.keras.datasets.cifar10.load_data()

        # one-hot-encode the labels
        training_labels = tf.keras.utils. \
            to_categorical(training_labels, num_classes)
        test_label = tf.keras.utils. \
            to_categorical(test_label, num_classes)

        training_data_generator = ImageDataGenerator(rescale=1. / 255)
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)

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

    # not working? blob are can only be created as shape (samples, features)
    elif dataset_parameters['dataset_name'] == 'blob':
        training_data, training_labels = make_blobs(n_samples=1000,
                                                    n_features=3,
                                                    centers=num_classes,
                                                    random_state=0)
        test_data, test_label = make_blobs(n_samples=100,
                                           n_features=3,
                                           centers=num_classes,
                                           random_state=0)

        training_labels = tf.keras.utils.to_categorical(training_labels,
                                                        num_classes)
        test_label = tf.keras.utils.to_categorical(test_label,
                                                   num_classes)

        training_generator = [training_data, training_labels]
        validation_generator = [test_data, test_label]

    return training_generator, validation_generator

