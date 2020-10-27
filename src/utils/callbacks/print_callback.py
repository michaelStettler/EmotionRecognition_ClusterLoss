import tensorflow as tf


class CustomPrintCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(CustomPrintCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print('\n** learning rate: {} **'.format(lr))
