import tensorflow as tf


class LossHistory(tf.keras.callbacks.Callback):

    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_accuracy'))


def save_metrics(history, metrics):
    metrics[0] = metrics[0] + history.losses
    metrics[1] = metrics[1] + history.accuracies
    metrics[2] = metrics[2] + history.val_losses
    metrics[3] = metrics[3] + history.val_accuracies
