import json
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np

from src.utils.model_utility_multi import load_model
from src.utils.generators import get_cluster_generator
from src.utils.data_collection import LossHistory
from src.utils.data_collection import save_metrics

from src.utils.callbacks.lr_callback import CustomLearningRateScheduler
from src.utils.callbacks.lr_callback import lr_schedule
from src.utils.callbacks.print_callback import CustomPrintCallback

from src.model_functions.WeightedSoftmaxCluster import SparseClusterLayer
from src.model_functions.WeightedSoftmaxCluster import WeightedClusterLoss


"""
python -m src.model_functions.train_model_cliuster_loss -m resnet50v2_clustLoss -d affectnet -c blue

"""

def train_model(model_configuration: str,
                dataset_configuration: str,
                computer_configuration: str):
    # loads name, image width/ height and l2_reg data
    with open('src/configuration/model/{}.json'
                      .format(model_configuration)) as json_file:
        model_parameters = json.load(json_file)

    # loads data, number of gpus
    with open('src/configuration/computer/{}.json'
                      .format(computer_configuration)) as json_file:
        computer_parameters = json.load(json_file)

    # loads n_classes, labels, class names, etc.
    with open('src/configuration/dataset/{}.json'
                      .format(dataset_configuration)) as json_file:
        dataset_parameters = json.load(json_file)

    # model template serves to save the model even with multi GPU training
    model = load_model(model_parameters,
                       dataset_parameters)

    # create the training and validation data
    training_data, validation_data = get_cluster_generator(dataset_parameters,
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

    # print('** classes indices: **', training_data.class_indices)
    class_weights = None
    if model_parameters['class_weights']:
        class_weights = {
            0: float(134414 / 24882),
            1: float(134414 / 3750),
            2: float(134414 / 3803),
            3: float(134414 / 6378),
            4: float(134414 / 134414),
            5: float(134414 / 74874),
            6: float(134414 / 25759),
            7: float(134414 / 14090),
        }
        print('** loaded class weights **', class_weights)

    # print(model.summary())
    # test generator
    # next(training_data)

    model.fit(training_data,
              epochs=model_parameters['number_epochs'],
              validation_data=validation_data,
              validation_steps=128,
              callbacks=callbacks_list,
              steps_per_epoch=2876507/model_parameters['batch_size']
              # class_weight=class_weights,
              # workers=12
              )
    #
    # save_metrics(history, metrics)
    #
    # evaluation = model.evaluate(validation_data,
    #                             # workers=12,
    #                             verbose=1)
    #
    # print("evaluation", evaluation)
    #
    # weight_path = '../weights/{}'.format(dataset_parameters['dataset_name'])
    # if not os.path.exists(weight_path):
    #     os.mkdir(weight_path)
    # metric_path = '../metrics/{}'.format(dataset_parameters['dataset_name'])
    # if not os.path.exists(metric_path):
    #     os.mkdir(metric_path)
    #
    # model.save(weight_path + '/{}_{}_{}_{}'.format(
    #     model_configuration,
    #     dataset_configuration,
    #     computer_configuration,
    #     datetime.now().strftime("%Y-%m-%d_%H-%M")))
    #
    # np.save(metric_path + '/{}_{}_{}_{}'.format(
    #     model_configuration,
    #     dataset_configuration,
    #     computer_configuration,
    #     datetime.now().strftime("%Y-%m-%d_%H-%M")), metrics)


if __name__ == '__main__':
    # runtime initialization will not allocate all memory on the device
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
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
