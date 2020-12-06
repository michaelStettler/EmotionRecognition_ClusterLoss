import sys
import tensorflow as tf
from argparse import ArgumentParser

from src.model_functions.train_model import *


if __name__ == '__main__':

    # runtime initialization will not allocate all memory on the device
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('** set memory growth **')
    except:
        print('Invalid device or cannot modify ' +
              'virtual devices once initialized.')
        pass

    parser = ArgumentParser()
    parser.add_argument("-m1", "--model1",
                        help="select your model")
    parser.add_argument("-m2", "--model2",
                        help="select your model",
                        default=None)
    parser.add_argument("-m3", "--model3",
                        help="select your model",
                        default=None)
    parser.add_argument("-m4", "--model4",
                        help="select your model",
                        default=None)
    parser.add_argument("-d", "--dataset",
                        help="select your dataset")
    parser.add_argument("-c", "--computer",
                        help="select your dataset")

    args = parser.parse_args()
    model1_configuration_name = args.model1
    model2_configuration_name = args.model2
    model3_configuration_name = args.model3
    model4_configuration_name = args.model4
    dataset_configuration_name = args.dataset
    computer_configuration_name = args.computer

    train_model(model1_configuration_name,
                dataset_configuration_name,
                computer_configuration_name)
    if model2_configuration_name:
        train_model(model2_configuration_name,
                    dataset_configuration_name,
                    computer_configuration_name)
    if model3_configuration_name:
        train_model(model3_configuration_name,
                    dataset_configuration_name,
                    computer_configuration_name)
    if model4_configuration_name:
        train_model(model4_configuration_name,
                    dataset_configuration_name,
                    computer_configuration_name)
