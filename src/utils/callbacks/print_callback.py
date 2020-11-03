import tensorflow as tf


class CustomPrintCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(CustomPrintCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        print('lr at Epoch {}: {}'.format(
            epoch,
            self.model.optimizer._decayed_lr(tf.float32).numpy()
        ))
