"""
Script to predict pre-trained models

run full training: python3 predict_model.py -m resnet18 -d affectnet_small -g 0,1,2 -c b -v 6 -da 2 -r 01
run chop off: python3 train_model.py -m vgg16 -d affectnet_one_batch -c m -mode tl -da 1 -r 01

"""

import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras import backend as K
from sklearn.utils import class_weight
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation, MaxPooling2D
from sklearn.metrics import classification_report

import sys
import os.path
import glob
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import time
import math

sys.path.insert(0, '../utils')
from parameters import *
from utils import *
from model_utils import *
from dataset import *

print("keras version:", keras.__version__)
print("floatx", keras.backend.floatx())


def predict_model(model_name='Simple',
                dataset='monkey_2',
                weights='imagenet',
                computer='a',
                run='00',
                task='classification',
                da='0',
                class_weights=None,
                version='0',
                save=False):
    print("coucou :)")

    # load the hyper parameters
    model_params = load_model_params(model_name, version, class_weights)
    computer = load_computer_params(computer, model_params)
    data = load_dataset_params(dataset, model_params, computer)
    # model template serves to save the model even with multi GPU training (use for training)
    model, model_template = load_model(True, data['weights_path'] + weights, model_params, data)

    loss = load_loss(data['loss_type'], model_params)

    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(lr=model_params['lr'][0], momentum=0.9, nesterov=False),
                  metrics=['mae', 'accuracy'])

    print("data loading")
    data_start_time = time.time()
    validation_generator = get_generator(data, model_params, da, task, validation_only=True)
    print("done loading data (%.2fs)" % (time.time() - data_start_time))
    print()

    # predict the model
    print("predict model")
    predictions = model.predict_generator(validation_generator,
                                         workers=12,
                                         steps=data['num_val_images']/model_params['batch_size'],
                                         # use_multiprocessing=True,
                                         verbose=1)

    print("shape prediction", np.shape(predictions))
    if 'affectnet' in data['dataset']:
        print(classification_report(validation_generator.classes,
                                    predictions.argmax(axis=1),
                                    target_names=data['class_names']))
    else:
        print(classification_report(validation_generator.classes,
                                    predictions.argmax(axis=1)))

    if save:
        np.save(data['weights_path'] + 'predictions', predictions)

    return predictions


if __name__ == '__main__':
    weights = 'imagenet'  # needed only for transfer learning
    # record starting time
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default='simple',
                        help="select which model to use: 'resnet18', 'resnet50', 'vgg16', 'simple'")
    parser.add_argument("-d", "--dataset",
                        default='imagenet',
                        help="select which dataset to train on: 'imagenet', 'affectnet', 'test', 'monkey'")
    parser.add_argument("-c", "--computer",
                        default='a',
                        help="select computer. a:980ti, b:2x1080ti, c.cluster")
    parser.add_argument("-r", "--run",
                        default='00',
                        help="set the run number")
    parser.add_argument("-g", "--gpu",
                        default='0',
                        help="set the gpu to use")
    parser.add_argument("-v", "--version",
                        default='0',
                        help="set the version to use")
    parser.add_argument("-t", "--task",
                        default='classification',
                        help="select the kind of learning, classification or regression")
    parser.add_argument("-da", "--data_augmentation",
                        default='2',
                        help="select which data augmentation to perform")
    parser.add_argument("-cw", "--class_weights",
                        default=None,
                        help="select which class weights to set into the weighted loss")
    parser.add_argument("-w", "--weights",
                        default=None,
                        help="name of the weights to load")
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    computer = args.computer
    run = args.run
    gpus = args.gpu
    version = args.version
    task = args.task
    da = args.data_augmentation
    cw = args.class_weights
    weights = args.weights

    if weights is not None:
        model_name, dataset, run, version, task, da, cw = get_args_from_weights(weights)
    else:
        if cw is not None:
            weights = '%s_%s_%s_da-%s_v-%s_cw-%s_%s.h5' % (
                model_name, task, dataset, da, version, cw, run)
        else:
            weights = '%s_%s_%s_da-%s_v-%s_%s.h5' % (
                model_name, task, dataset, da, version, run)

    print("------------------------------------------------------------")
    print("                       Summary                              ")
    print()
    print("model:   ", model_name, " dataset:", dataset, "  task:", task)
    print("computer:", computer, "        run:", run, "  gpu:", gpus, "  version:", version, "  da:", da)
    print("Weights:", weights)
    print()
    print("------------------------------------------------------------")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    predict_model(model_name=model_name,
                dataset=dataset,
                weights=weights,
                computer=computer,
                run=run,
                task=task,
                da=da,
                version=version,
                  class_weights=cw,
                  save=True)

    print()
    print()
    print("------------------------------------------------------------")
    print("                    Summary End (%.2fs)                      " % (time.time() - start))
    print()
    print("model:   ", model_name, " dataset:", dataset, "  task:", task)
    print("computer:", computer, "        run:", run, "  gpu:", gpus, "  version:", version, "  da:", da)
    print("Weights:", weights)
    print()
    print("------------------------------------------------------------")
    print()
