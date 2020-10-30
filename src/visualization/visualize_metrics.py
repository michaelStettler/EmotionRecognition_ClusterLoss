import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def plot_metrics(metrics_path: str, plot_name: str, path_name: str):
    path = 'C:/Users/janni/Desktop/'
    if not os.path.exists(path + path_name):
        os.mkdir(path + path_name)

    # load the metrics values
    metrics = np.load(metrics_path, allow_pickle=True)
    losses = metrics[0]
    accuracies = metrics[1]
    val_losses = metrics[2]
    val_accuracies = metrics[3]

    num_epoch = np.shape(val_losses)[0]
    num_batch_per_epoch = np.shape(losses)[0] // num_epoch

    val_losses_batch = []
    val_accuracies_batch = []
    for epoch_count in range(num_epoch):
        for batch_count in range(num_batch_per_epoch):
            val_losses_batch.append(val_losses[epoch_count])
            val_accuracies_batch.append(val_accuracies[epoch_count])

    losses_epoch = []
    accuracies_epoch = []

    for i, loss in enumerate(losses):
        if i % num_batch_per_epoch == 0:
            losses_epoch.append(loss)
            accuracies_epoch.append(accuracies[i])

    size = np.shape(losses_epoch)[0]
    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, size), losses_epoch, label="train_loss")
    plt.plot(np.arange(0, size), val_losses, label="val_loss")
    plt.plot(np.arange(0, size), accuracies_epoch, label="acc")
    plt.plot(np.arange(0, size), val_accuracies, label="val_acc")
    plt.title("Training Loss and Accuracy for {}".format(plot_name))
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig(path + path_name + '/acc_loss_combined.png',
                bbox_inches='tight')

    # plot accuracy and loss with different y-axes
    fig, ax1 = plt.subplots()
    color = 'red'
    color2 = 'magenta'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(np.arange(0, size), losses_epoch, label="train_loss", color=color)
    ax1.plot(np.arange(0, size), val_losses, label="val_loss", color=color2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid()
    ax2 = ax1.twinx()
    color = 'darkblue'
    color2 = 'black'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(np.arange(0, size), accuracies_epoch, label="acc", color=color)
    ax2.plot(np.arange(0, size), val_accuracies, label="val_acc", color=color2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.title("Training Loss and Accuracy for {}".format(plot_name))
    # plt.show()
    plt.savefig(path + path_name + '/acc_loss_separated.png',
                bbox_inches='tight')

    # plot accuracy only
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, size), accuracies_epoch, label="acc")
    plt.plot(np.arange(0, size), val_accuracies, label="val_acc")
    plt.title("Accuracy for {}".format(plot_name))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig(path + path_name + '/acc_epoch.png',
                bbox_inches='tight')

    # plot accuracy only
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, size), losses_epoch, label="train_loss")
    plt.plot(np.arange(0, size), val_losses, label="val_loss")
    plt.title("Loss for {}".format(plot_name))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(path + path_name + '/loss_epoch.png',
                bbox_inches='tight')

    # plot accuracy over every batch
    size = np.shape(losses)[0]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, size), accuracies, label="acc")
    plt.plot(np.arange(0, size), val_accuracies_batch, label="val_acc")
    plt.title("Accuracy for {}".format(plot_name))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig(path + path_name + '/acc_batch.png',
                bbox_inches='tight')

    # plot loss over every batch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, size), losses, label="train_loss")
    plt.plot(np.arange(0, size), val_losses_batch, label="val_loss")
    plt.title("Loss for {}".format(plot_name))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(path + path_name + '/loss_batch.png',
                bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to metric")
    parser.add_argument("-n", "--name",
                        help="name for the plot")
    parser.add_argument("-pn", "--pathname",
                        help="name for the plot")

    args = parser.parse_args()
    metric_path = args.path
    name = args.name
    path_name = args.pathname

    plot_metrics(metric_path, name, path_name)
