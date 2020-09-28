"""
Plot the histogram accuracies of the trained models
USE: python3 plot_hist_accuracies.py -m resnet18 -d affectnet -r 01 -v 1

"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import keras

from argparse import ArgumentParser
sys.path.insert(0, '../utils')
from parameters import *
from tqdm import tqdm

np.set_printoptions(precision=2, linewidth=200, suppress=True)
train = False
one_batch = True  # just for coding purpose as it is process faster


def get_cv2_preprocess_img(img_name, model_params):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (model_params['img_width'], model_params['img_width']))
    x = np.expand_dims(x, axis=0)
    return x / 255


def get_auto_generator(data, model_params, data_augmentation):
    train_data_dir = data['train_dir']
    validation_data_dir = data['val_dir']

    if not os.path.isdir(train_data_dir):
        raise ValueError('Train directory does not exist')
    if not os.path.isdir(validation_data_dir):
        raise ValueError('Validation directory does not exist')

    if train:
        data_dir = train_data_dir
    else:
        data_dir = validation_data_dir

    # get the data generator
    datagen = get_datagen(data_augmentation, data, train=False)

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(model_params['img_height'], model_params['img_width']),
        batch_size=model_params['batch_size'],
        class_mode="categorical",
        shuffle=False,
        #save_to_dir='test_val'
    )

    return generator


def get_csv_generator(data, model_params, data_augmentation, task):
    if train:
        # df_val = pd.read_csv(data['csv_train_file'], nrows=1)
        df_val = pd.read_csv(data['csv_train_file'])
    else:
        # df_val = pd.read_csv(data['data_path'] + 'validation_modified.csv')
        df_val = pd.read_csv(data['csv_val_file'])
    # get the data generator (from parameters.py)
    val_datagen = get_datagen(data_augmentation, data, train=False)

    if task == 'classification':
        class_mode = 'categorical'
        y_col = data['class_label']
    elif task == 'regression':
        class_mode = 'other'
        y_col = data['box_labels']
    else:
        raise ValueError('Wrong task selected - %s doest not exist' % task)

    if train:
        dir_name = data['train_dir']
    else:
        dir_name = data['val_dir']

    generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=dir_name,
        x_col='subDirectory_filePath',
        y_col=y_col,
        has_ext=True,
        class_mode=class_mode,
        # class_mode=None,
        target_size=(model_params['img_height'], model_params['img_width']),
        batch_size=model_params['batch_size'],
        shuffle=False,
        # save_to_dir='test_val'
    )

    # print("generator")
    # print(np.shape(generator))

    return generator, df_val


def plot_accuracies(model_name, version, computer, dataset, da, run, task):
    # load the hyper parameters
    model_params = load_model_params(model_name, version)
    computer = load_computer_params(computer, model_params)

    if mode == 'tl':
        print("need to do tl version")
    else:
        weights_path = '%sweights/%s_%s_%s_da-%s_v-%s_%s.h5' % (model_params['model_path'], model_name, task, dataset, da, version, run)

    print(weights_path)
    if not os.path.isfile(weights_path):
        print("weights file does not exists!")
        return

    if one_batch:
        dataset = 'affectnet_one_batch'
    data = load_dataset_params(dataset, model_params, computer)

    # load the model
    model, model_template = load_model(True, weights_path, model_params, data)
    # model.summary()

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
                  metrics=['mae', 'accuracy'])

    # load the data
    if data['labels_type'] == 'csv':
        generator, df_val = get_csv_generator(data, model_params, da, task)
    else:
        generator = get_auto_generator(data, model_params, da)

    # -----------------------------------------------------------------------------#
    #                                       keras methods
    # -----------------------------------------------------------------------------#
    # evaluate the model
    print("predict")
    # predictions = model.predict_generator(generator, use_multiprocessing=False, workers=1, max_queue_size=1, verbose=1)
    predictions = model.predict_generator(generator, verbose=1)
    print("evaluate")
    # score = model.evaluate_generator(generator, workers=1, max_queue_size=1, use_multiprocessing=False, verbose=1)
    score = model.evaluate_generator(generator, verbose=1)
    print("score")
    print(model.metrics_names)
    print(score)
    print()

    argmax = np.argmax(predictions, axis=1)
    count = []
    for i in range(11):
        count.append(np.sum(argmax == i))
    print("keras predicitons")
    print(count, np.sum(count))
    print()

    # -----------------------------------------------------------------------------#
    #                               hand made methods
    # -----------------------------------------------------------------------------#
    # counting right predictions
    counting_true = np.zeros(data['n_classes'])
    counting_pred = np.zeros(data['n_classes'])
    counting_tot = np.zeros(data['n_classes'])

    # dico = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2]  # auto generator
    dico = [1, 0, 6, 7, 10, 8, 3, 9, 2, 4, 5]  # csv
    for i, line in enumerate(tqdm(df_val.iterrows())):
        # get image name
        img_name = df_val.loc[i, 'subDirectory_filePath']
        # get image label
        # the label goes to the dico map, this step hab been made by hand and counteract the automatic labelling
        # from keras
        true_val = dico[df_val.loc[i, data['class_label']]]
        # load image and pre-process it for the network
        if train:
            x = get_cv2_preprocess_img(data['train_dir'] + img_name, model_params)
        else:
            x = get_cv2_preprocess_img(data['val_dir'] + img_name, model_params)

        # send img through the network
        pred = model.predict(x)

        # count the correctly predicted images
        if np.argmax(pred) == true_val:
            counting_true[true_val] += 1

        counting_pred[np.argmax(pred)] += 1
        counting_tot[true_val] += 1

    # getting classes accuracy
    classes_accuracies = np.zeros(data['n_classes'])
    for i in range(data['n_classes']):
        classes_accuracies[i] = counting_true[i] / counting_tot[i] * 100

    print("counting")
    print(counting_true, np.sum(counting_true))
    print(counting_pred, np.sum(counting_pred))
    print(counting_tot, np.sum(counting_tot))
    print(classes_accuracies)
    print()

    # control accuracy
    accuracy = 0
    tot_correct = np.sum(counting_true)
    tot_img = np.sum(counting_tot)
    for i in range(data['n_classes']):
        accuracy += classes_accuracies[i] * counting_tot[i] / tot_img
        # print(np.sum(classes_accuracies / data['n_classes']))

    print("Calculated accuracy", accuracy)
    print("tot_correct/tot_img = %.0f/%.0f = %.4f" % (tot_correct, tot_img, tot_correct / tot_img * 100), "%")
    print()
    # -----------------------------------------------------------------------------#

    names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'No-Face']
    fig, ax = plt.subplots(1)
    ax.bar(names, classes_accuracies)
    ax.set_title("class accuracies")
    ymin, ymax = 0, 100
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("classes names")
    ax.set_ylabel("% accuracy")
    plt.show()


if __name__ == '__main__':
    weights = 'imagenet'  # needed only for transfer learning
    task = 'classification'  # classification regression

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
                        help="select which data augmentation to perform")
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

    plot_accuracies(model_name, version, computer, dataset, da, run, task)
