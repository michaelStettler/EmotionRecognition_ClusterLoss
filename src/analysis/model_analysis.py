import os
import sys
from argparse import ArgumentParser
import time
import numpy as np
import six

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../visualization')
from model_utils import *
from predict_model import *
from evaluate_model import *
from plot_metrics import *


def analyse_model(model_name='Simple',
                  dataset='monkey_2',
                  weights=None,
                  computer='a',
                  run='00',
                  mode='full',
                  task='classification',
                  da='0',
                  class_weights=None,
                  version='0'):

    # load params
    model_params = load_model_params(model_name, version, class_weights)
    computer_params = load_computer_params(computer, model_params)
    data = load_dataset_params(dataset, model_params, computer_params)

    # get validation data_processing
    print("data_processing loading")
    data_start_time = time.time()
    validation_generator = get_generator(data, model_params, da, task, validation_only=True)
    print("done loading data_processing (%.2fs)" % (time.time() - data_start_time))
    print()

    # get training metrics
    print("metrics")
    if mode == 'tl':
        print("need to do tl version")
    else:
        print("todo take care of imagenet pre trained weights ?")
        metrics_path = model_params['model_path'] + 'metrics/metrics_' + weights[:-2] + 'npy'

    losses_epoch, val_losses, accuracies_epoch, val_accuracies = plot_metrics(metrics_path, weights)
    #
    # get evaluation metrics
    print("evaluation")
    evaluation = evaluate_model(model_name, dataset, weights, computer, run, task, da, class_weights, version)

    # get prediction metrics
    print("prediction")
    predictions = predict_model(model_name, dataset, weights, computer, run, task, da, class_weights, version)

    if 'affectnet' in data['dataset']:
        class_report = classification_report(validation_generator.classes,
                                             predictions.argmax(axis=1),
                                             target_names=data['class_names'])
    else:
        print(classification_report(validation_generator.classes,
                                    predictions.argmax(axis=1)))
    # get confusion matrix
    conf_mat = confusion_matrix(validation_generator.classes, predictions.argmax(axis=1))

    # plot results into one graph
    plt.style.use("ggplot")
    f = plt.figure(figsize=(12, 14))
    f.suptitle("Model Analysis")

    # create the layout
    gs = gridspec.GridSpec(5, 4)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1:3, :2])
    ax3 = plt.subplot(gs[1:3, 2:])
    ax4 = plt.subplot(gs[3:5, :3])

    ax1.set_facecolor('white')
    t0 = 'Architecture: %s' % model_name
    t1 = 'Dataset:         %s' % dataset
    t2 = 'Version:          %s' % version
    t3 = 'Run:            %s' % run
    t4 = 'Data augm: %s' % da
    t5 = 'Mode:          %s' % mode
    t6 = 'Saved Acc:  %.4f' % val_accuracies[-1]
    t7 = 'Pred Acc:     %.4f' % evaluation[2]
    t8 = 'Saved Loss: %.4f' % val_losses[-1]
    t9 = 'Pred Loss:    %.4f' % evaluation[0]
    t10 = 'lr: %s' % model_params['lr']
    t11 = 'num epochs: %s' % model_params['num_epochs']
    t12 = 'l2-reg: %s' % model_params['l2_reg']
    ax1.text(0.1, .9, t0, ha='left')
    ax1.text(0.1, .7, t1, ha='left')
    ax1.text(0.1, .5, t2, ha='left')
    ax1.text(0.4, .9, t3, ha='left')
    ax1.text(0.4, .7, t4, ha='left')
    ax1.text(0.4, .5, t5, ha='left')
    ax1.text(0.7, .9, t6, ha='left')
    ax1.text(0.7, .8, t7, ha='left')
    ax1.text(0.7, .6, t8, ha='left')
    ax1.text(0.7, .5, t9, ha='left')
    ax1.text(0.1, .2, t10, ha='left')
    ax1.text(0.1, .1, t11, ha='left')
    ax1.text(0.4, .2, t12, ha='left')
    ax1.tick_params(labelbottom=False, labelleft=False, color='white')

    size = np.shape(losses_epoch)[0]
    ax2.plot(np.arange(0, size), losses_epoch, label="train_loss")
    ax2.plot(np.arange(0, size), val_losses, label="val_loss")
    ax2.plot(np.arange(0, size), accuracies_epoch, label="acc")
    ax2.plot(np.arange(0, size), val_accuracies, label="val_acc")
    ax2.set_xlabel("Epoch #")
    ax2.set_ylabel("Loss/Accuracy")
    ax2.legend()

    ax3.set_facecolor('white')
    ax3.text(0.1, 0.2, class_report)
    ax3.tick_params(labelbottom=False, labelleft=False, color='white')

    ax4.set_facecolor('white')
    ax4.axis('off')
    cell_text = np.zeros(np.shape(conf_mat), dtype=int)
    for row in range(np.shape(conf_mat)[0]):
        for column in range(np.shape(conf_mat)[1]):
            cell_text[row, column] = conf_mat[row, column]

    if 'affectnet' in data['dataset']:
        conf_table = ax4.table(cellText=cell_text,
                               rowLabels=data['class_names'],
                               colLabels=data['class_names'],
                               loc='center')
    else:
        conf_table = ax4.table(cellText=cell_text,
                               loc='center')

    for k, cell in six.iteritems(conf_table._cells):
        cell.set_edgecolor('white')
    ax4.tick_params(labelbottom=False, labelleft=False, color='white')

    plt.show()


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
    parser.add_argument("-mode", "--mode",
                        default='full',
                        help="select if train all (full) or use transfer learning (tl)")
    parser.add_argument("-v", "--version",
                        default='0',
                        help="set the version to use")
    parser.add_argument("-t", "--task",
                        default='classification',
                        help="select the kind of learning, classification or regression")
    parser.add_argument("-da", "--data_augmentation",
                        default='0',
                        help="select which data_processing augmentation to perform")
    parser.add_argument("-cw", "--class_weights",
                        default=None,
                        help="select which class weights to set into the weighted loss")
    parser.add_argument("-w", "--weights",
                        default=None,
                        help="name of the weights to load")
    args = parser.parse_args()

    model_name = args.model
    mode = args.mode
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
    print("weights:", weights)
    print()
    print("------------------------------------------------------------")
    print()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    analyse_model(model_name=model_name,
                  dataset=dataset,
                  weights=weights,
                  computer=computer,
                  run=run,
                  task=task,
                  da=da,
                  class_weights=cw,
                  version=version)

    print()
    print()
    print("------------------------------------------------------------")
    print("                    Summary End (%.2fs)                      " % (time.time() - start))
    print()
    print("model:   ", model_name, " dataset:", dataset, "  task:", task)
    print("computer:", computer, "        run:", run, "  gpu:", gpus, "  version:", version, "  da:", da)
    print("weights:", weights)
    print()
    print("------------------------------------------------------------")
    print()