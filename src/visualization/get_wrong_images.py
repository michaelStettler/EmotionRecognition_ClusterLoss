"""
Plot the histogram accuracies of the trained models
USE: python3 get_wrong_images.py -m resnet18 -d affectnet_one_batch -c m -r 01 -v 1

"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
import shutil

from argparse import ArgumentParser
sys.path.insert(0, '../utils')
from parameters import *

np.set_printoptions(precision=0, linewidth=200, suppress=True)


def get_csv_generator(data, model_params, data_augmentation, task):
    # always validate on the whole set
    # df_val = pd.read_csv(data_processing['data_path'] + 'validation_modified.csv')
    print("****************************")
    print("Need to change, only for testing purpose yet")
    print("****************************")
    df_val = pd.read_csv(data['data_path'] + 'validation_one_batch.csv')
    # get the data_processing generator (from parameters.py)
    val_datagen = get_datagen(data_augmentation, data, train=False)

    if task == 'classification':
        class_mode = 'categorical'
        y_col = data['class_label']
    elif task == 'regression':
        class_mode = 'other'
        y_col = data['box_labels']
    else:
        raise ValueError('Wrong task selected - %s doest not exist' % task)

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=data['data_path'] + 'validation_one_batch/',
        x_col='subDirectory_filePath',
        y_col=y_col,
        has_ext=True,
        class_mode=class_mode,
        target_size=(model_params['img_height'], model_params['img_width']),
        batch_size=model_params['batch_size'],
        shuffle=False,
        #save_to_dir='test_val'
    )

    return validation_generator, df_val


def get_wrong_images(model_name, version, computer, dataset, da, run, task):
    print("prout prout")

    # load the hyper parameters
    model_params = load_model_params(model_name, version)
    computer = load_computer_params(computer, model_params)
    data = load_dataset_params(dataset, model_params, computer)

    if mode == 'tl':
        print("need to do tl version")
    else:
        weights_path = '%sweights/%s_%s_%s_da-%s_v-%s_%s.h5' % (model_params['model_path'], model_name, task, dataset, da, version, run)

    print(weights_path)
    if not os.path.isfile(weights_path):
        print("weights file does not exists!")
        return

    # load the model
    model = load_model(True, weights_path, model_params, data)

    # define the loss
    def constrainedCrossEntropy(ytrue, ypred):
        ypred = keras.backend.clip(ypred, 0.0001, 0.9999)
        return keras.losses.categorical_crossentropy(ytrue, ypred)

    # loss = keras.losses.categorical_crossentropy
    loss = constrainedCrossEntropy
    if task == 'regression':
        loss = keras.losses.mean_squared_error
    # compile the model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(lr=model_params['lr'][0], momentum=0.9, nesterov=True),
                  metrics=['mae', 'categorical_accuracy'])

    # load the data_processing
    if 'AFFECTNET' in data['dataset'].upper():
        val_generator, df_val = get_csv_generator(data, model_params, da, task)
    else:
        print("need to do for imagenet")

    # evaluate the model
    predictions = model.predict_generator(val_generator, use_multiprocessing=True, verbose=1)

    # finding the right predictions
    for i, pred in enumerate(predictions):
        true_val = df_val.loc[i, data['class_label']]
        if np.argmax(pred) != true_val:
            img_name = df_val.loc[i, 'subDirectory_filePath']
            print("Images %s is wrong" % img_name)
            shutil.copyfile(data['val_dir'] + img_name, data['data_path'] + 'wrong_img/' + img_name)


if __name__ == '__main__':

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
    parser.add_argument("-mode", "--mode",
                        default='full',
                        help="select if train all or use transfer learning")
    parser.add_argument("-t", "--task",
                        default='classification',
                        help="select the kinf of learning, classification or regression")
    parser.add_argument("-da", "--data_augmentation",
                        default='2',
                        help="select which data_processing augmentation to perform")
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    computer = args.computer
    run = args.run
    gpus = args.gpu
    version = args.version
    mode = args.mode
    task = args.task
    da = args.data_augmentation

    print("loading:")
    print("model", model_name, "dataset", dataset, "mode", mode, "task", task)
    print("computer:", computer, "run", run, "gpu", gpus, "version", version)

    get_wrong_images(model_name, version, computer, dataset, da, run, task)