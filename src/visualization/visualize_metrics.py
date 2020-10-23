import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def plot_metrics(metrics_path):

    # load the metrics values
    metrics = np.load(metrics_path, allow_pickle=True)
    losses = metrics[0]
    accuracies = metrics[1]
    val_losses = metrics[2]
    val_accuracies = metrics[3]

    num_epoch = np.shape(val_losses)[0]
    num_batch_per_epoch = np.shape(losses)[0] // num_epoch

    losses_epoch = []
    accuracies_epoch = []
    print(accuracies)
    for i, loss in enumerate(losses):
        if i % num_batch_per_epoch == 0:
            losses_epoch.append(loss)
            accuracies_epoch.append(accuracies[i])

    size = np.shape(losses_epoch)[0]
    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    # plt.plot(np.arange(0, size), losses_epoch, label="train_loss")
    # plt.plot(np.arange(0, size), val_losses, label="val_loss")
    plt.plot(np.arange(0, size), accuracies_epoch, label="acc")
    plt.plot(np.arange(0, size), val_accuracies, label="val_acc")
    plt.title("Training Loss and Accuracy for")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    weights = 'imagenet'  # needed only for transfer learning

    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to metric")

    args = parser.parse_args()
    metric_path = args.path

    plot_metrics(metric_path)
