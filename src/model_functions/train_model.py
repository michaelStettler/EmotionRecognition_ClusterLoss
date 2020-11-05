import json
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np

sys.path.insert(0, '../utils')
from model_utility_multi import *
from generators import *
from data_collection import *

sys.path.insert(0, '../utils/callbacks')
from lr_callback import *
from print_callback import *


def train_model(model_configuration: str,
                dataset_configuration: str,
                computer_configuration: str):
    # loads name, image width/ height and l2_reg data
    with open('../configuration/model/{}.json'
                      .format(model_configuration)) as json_file:
        model_parameters = json.load(json_file)

    # loads data, number of gpus
    with open('../configuration/computer/{}.json'
                      .format(computer_configuration)) as json_file:
        computer_parameters = json.load(json_file)

    # loads n_classes, labels, class names, etc.
    with open('../configuration/dataset/{}.json'
                      .format(dataset_configuration)) as json_file:
        dataset_parameters = json.load(json_file)

    # model template serves to save the model even with multi GPU training
    model = load_model(model_parameters,
                       dataset_parameters)

    # create the training and validation data
    training_data, validation_data = get_generator(dataset_parameters,
                                                   model_parameters,
                                                   computer_parameters)

    callbacks_list = []
    history = LossHistory()
    metrics = [[], [], [], []]

    # add callbacks for training
    if model_parameters['lr_scheduler'][0]:
        print('** lr_scheduler enabled **')
        lr_scheduler = CustomLearningRateScheduler(
            model_parameters,
            lr_schedule,
            model_parameters['lr_scheduler'])
        callbacks_list.append(lr_scheduler)
    if model_parameters['early_stopping']:
        print('** early_stopping enabled **')
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=model_parameters['early_stopping_monitor'],
            patience=3,
            restore_best_weights=True)
        callbacks_list.append(early_stopping)

    callbacks_list.append(history)
    callbacks_list.append(CustomPrintCallback())

    class_weights = None
    if model_parameters['class_weights']:
        class_weights = {
            0: float(287650 / 74874),
            1: float(287650 / 134414),
            2: float(287650 / 25459),
            3: float(287650 / 14090),
            4: float(287650 / 6378),
            5: float(287650 / 3803),
            6: float(287650 / 24882),
            7: float(287650 / 3750),
        }

    model.fit(training_data,
              epochs=model_parameters['number_epochs'],
              validation_data=validation_data,
              validation_steps=128,
              callbacks=callbacks_list,
              class_weights=class_weights,
              workers=12)

    save_metrics(history, metrics)

    evaluation = model.evaluate(validation_data,
                                workers=12,
                                verbose=1)

    print("evaluation", evaluation)

    weight_path = '../weights/{}'.format(dataset_parameters['dataset_name'])
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    metric_path = '../metrics/{}'.format(dataset_parameters['dataset_name'])
    if not os.path.exists(metric_path):
        os.mkdir(metric_path)

    model.save(weight_path + '/{}_{}_{}_{}'.format(
        model_configuration,
        dataset_configuration,
        computer_configuration,
        datetime.now().strftime("%Y-%m-%d_%H-%M")))

    np.save(metric_path + '/{}_{}_{}_{}'.format(
        model_configuration,
        dataset_configuration,
        computer_configuration,
        datetime.now().strftime("%Y-%m-%d_%H-%M")), metrics)


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
    parser.add_argument("-m", "--model",
                        help="select your model")
    parser.add_argument("-d", "--dataset",
                        help="select your dataset")
    parser.add_argument("-c", "--computer",
                        help="select your dataset")

    args = parser.parse_args()
    model_configuration_name = args.model
    dataset_configuration_name = args.dataset
    computer_configuration_name = args.computer

    train_model(model_configuration_name,
                dataset_configuration_name,
                computer_configuration_name)
