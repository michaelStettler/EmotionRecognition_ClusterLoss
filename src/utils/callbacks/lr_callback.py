import tensorflow as tf


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according
    to schedule.

    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, model_parameters,
                 schedule,
                 learning_rates: [(int, float)]):

        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.learning_rates = learning_rates
        self.model_parameters = model_parameters

    def on_epoch_begin(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(lr)
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.learning_rates)

        if self.model_parameters['lr_decay']:
            scheduled_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=scheduled_lr,
                decay_steps=self.model_parameters['lr_decay_steps'],
                decay_rate=self.model_parameters['lr_decay_rate'])

        # Set the value back to the optimizer before this epoch starts
        self.model.optimizer.lr = scheduled_lr


def lr_schedule(epoch, lr, learning_rates: [(int, float)]):
    """Helper function to retrieve the scheduled learning rate
    based on epoch."""

    if epoch < learning_rates[0][0] or epoch > learning_rates[-1][0]:
        return lr
    for i in range(len(learning_rates)):
        if epoch == learning_rates[i][0]:
            return learning_rates[i][1]
    return lr
