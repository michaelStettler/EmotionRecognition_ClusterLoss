import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from tensorflow.keras import datasets
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.python.keras.utils.vis_utils import plot_model

"""
Center loss: https://ydwen.github.io/papers/WenECCV16.pdf
Helped taken from:
 
https://github.com/zoli333/Center-Loss
https://github.com/Kakoedlinnoeslovo/center_loss/blob/master/Network.py
"""

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# load data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print("shape train_images", np.shape(train_images))
print("shape train_labels", np.shape(train_labels))
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
num_classes = 10
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# create fake clusters to pass as argument
train_cl = np.zeros((train_images.shape[0], 1))
test_cl = np.zeros((test_images.shape[0], 1))

# prepare train and test sets
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')


class CenterLayer(Layer):
    def __init__(self, num_classes, alpha_center, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.alpha_center = alpha_center

    def build(self, input_shape):
        # split input
        features = input_shape[0]

        # Create a trainable weight variable for this layer
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_classes, features[-1]),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x):
        # split data
        y_pred = x[0]
        y_true = x[1]

        # transform to one hot encoding
        y_true = tf.cast(y_true, dtype=tf.uint8)
        y_true = tf.one_hot(y_true, self.num_classes)
        y_true = tf.cast(y_true, dtype='float32')
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], self.num_classes))

        # compute center loss
        delta_centers = K.dot(tf.transpose(y_true), (K.dot(y_true, self.centers) - y_pred))
        denominator = K.sum(tf.transpose(y_true), axis=1, keepdims=True) + 1
        delta_centers /= denominator
        new_centers = self.centers - self.alpha_center * delta_centers
        self.add_update((self.centers, new_centers))
        result = (K.dot(y_true, self.centers) - y_pred)
        return K.sum(result ** 2, axis=1, keepdims=True)


# ----------------------- create model --------------------------------
input = tf.keras.Input(shape=(28, 28, 1), name="base_input")
label = tf.keras.Input(shape=(1,), name="labels", dtype='int32')
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(10, name='output')(x)
cluster = CenterLayer(num_classes=10, alpha_center=0.5, name='cluster')([x, label])

model = tf.keras.Model(inputs=[input, label], outputs=[output, cluster])


def cluster_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss={'output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    'cluster': cluster_loss},
              loss_weights=[1, .5],
              metrics={'output': ['accuracy']},
              class_weights=[class_weights, class_weights])
# model.fit(train_images, train_labels, epochs=10,
# #           validation_data=(test_images, test_labels))

# model.train_on_batch([train_images[:32], train_labels[:32]], [train_labels[:32], train_cl[:32]])

model.fit([train_images, train_labels], [train_labels, train_cl], epochs=20,
          validation_data=([test_images, test_labels], [test_labels, test_cl]))

