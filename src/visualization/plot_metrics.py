"""
    Plot the save metrics of the trained models
    Run: python3 plot_metrics.py -m resnet18 -d affectnet_one_batch -da 2 -v 1 -r 01

"""

import numpy as np
import sys
import matplotlib.pyplot as plt

from argparse import ArgumentParser
sys.path.insert(0, '../utils')
from parameters import *
from utils import *


def plot_metrics(metrics_path, title, show=False):
    print("mertrics path", metrics_path)

    # load the metrics values
    metrics = np.load(metrics_path)
    losses = metrics[0]
    accuracies = metrics[1]
    # accuracies = np.multiply(metrics[1], 100)  # convert to %
    val_losses = metrics[2]
    val_accuracies = metrics[3]
    # val_accuracies = np.multiply(metrics[3], 100)  # convert to %

    # keep only the training loss at the end of the epoch
    # as to math the validation set -> too simple but for now ok
    # we should either plot all of them or make the average per epoch
    print()
    print("****************************************************************")
    print()
    print("reminder to see that comment if we want to show on of this graph")
    print()
    print("****************************************************************")
    print()
    num_epoch = np.shape(val_losses)[0]
    num_batch_per_epoch = np.shape(losses)[0] // num_epoch
    losses_epoch = []
    accuracies_epoch = []
    for i, loss in enumerate(losses):
        if i % num_batch_per_epoch == 0:
            losses_epoch.append(loss)
            accuracies_epoch.append(accuracies[i])

    # print out the last numbers
    print("final loss training: %.4f, final loss validation: %.4f" % (losses_epoch[-1], val_losses[-1]))
    print("final accuracy training: %.4f, final accuracy validation: %.4f" % (accuracies_epoch[-1], val_accuracies[-1]))

    # # create the figures
    # # first figure
    # plt.figure()
    # plt.title("Losses of %s" % model_name)
    # plt.plot(losses_epoch, label='training')
    # plt.plot(val_losses, label='validation')
    # plt.xlabel('# epochs')
    # plt.ylabel('losses')
    # plt.legend()
    #
    # # second figure
    # plt.figure()
    # plt.title("Accuracies of %s" % model_name)
    # plt.plot(accuracies_epoch, label='training')
    # plt.plot(val_accuracies, label='validation')
    # plt.xlabel('# epochs')
    # plt.ylabel('% accuracy')
    # plt.ylim((0, 100))
    # plt.legend()

    # show
    if show:
        size = np.shape(losses_epoch)[0]
        # plot the training + testing loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, size), losses_epoch, label="train_loss")
        plt.plot(np.arange(0, size), val_losses, label="val_loss")
        plt.plot(np.arange(0, size), accuracies_epoch, label="acc")
        plt.plot(np.arange(0, size), val_accuracies, label="val_acc")
        plt.title("Training Loss and Accuracy for \n %s" % title)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

    return losses_epoch, val_losses, accuracies_epoch, val_accuracies


if __name__ == '__main__':
    weights = 'imagenet'  # needed only for transfer learning

    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default='simple',
                        help="select which model to use: 'resnet18', 'resnet50', 'vgg16', 'simple'")
    parser.add_argument("-d", "--dataset",
                        default='imagenet',
                        help="select which dataset to train on: 'imagenet', 'affectnet', 'test', 'monkey'")
    parser.add_argument("-r", "--run",
                        default='00',
                        help="set the run number")
    parser.add_argument("-v", "--version",
                        default='0',
                        help="set the version to use")
    parser.add_argument("-mode", "--mode",
                        default='full',
                        help="select if train all or use transfer learning")
    parser.add_argument("-t", "--task",
                        default='classification',
                        help="select the kind of learning, classification or regression")
    parser.add_argument("-da", "--data_augmentation",
                        default='2',
                        help="select which data augmentation to perform")
    parser.add_argument("-w", "--weights",
                        default=None,
                        help="name of the weights to load")
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    run = args.run
    version = args.version
    mode = args.mode
    task = args.task
    da = args.data_augmentation
    weights = args.weights

    if weights is not None:
        model_name, dataset, run, version, task, da = get_args_from_weights(weights)
        title = weights
    else:
        title = '%s_%s_%s_da-%s_v-%s_%s.h5' % (model_name, task, dataset, da, version, run)

    print("------------------------------------------------------------")
    print("                       Summary                              ")
    print()
    print("model:   ", model_name, " dataset:", dataset, "  task:", task)
    print("run:", run, "version:", version, "  da:", da)
    if weights is not None:
        print("Weights:", weights)
    print()
    print("------------------------------------------------------------")
    print()

    # load the hyper parameters
    model_params = load_model_params(model_name, version)

    if mode == 'tl':
        print("need to do tl version")
    else:
        if weights is not None:
            metrics_path = model_params['model_path'] + 'metrics/metrics_' + weights[:-2] + 'npy'
        else:
            metrics_path = '%smetrics/metrics_%s_%s_%s_da-%s_v-%s_%s.npy' % (model_params['model_path'], model_name, task, dataset, da, version, run)

    fig = plot_metrics(metrics_path, title, show=True)
