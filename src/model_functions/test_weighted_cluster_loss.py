import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from tensorflow.keras import datasets

from src.model_functions.WeightedSoftmaxCluster import SparseClusterLayer
from src.model_functions.WeightedSoftmaxCluster import SparseWeightedSoftmaxLoss
from src.model_functions.WeightedSoftmaxCluster import SparseWeightedSoftmaxLoss2
from src.model_functions.WeightedSoftmaxCluster import WeightedClusterLoss

from tensorflow.python.keras.utils.vis_utils import plot_model

"""
Test of the Weighted Softmax Cluster layer and loss function on Cifar10

Weighted Softmax-Cluster loss: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7249188/
  
"""

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print("shape train_images", np.shape(train_images))
print("shape train_labels", np.shape(train_labels))
num_classes = 10
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# prepare train and test sets
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')


# ----------------------- create model --------------------------------
input = tf.keras.Input(shape=(32, 32, 3), name="base_input")
labels = tf.keras.Input(shape=(1, ), dtype='int32')
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(10, name='output')(x)
scl = SparseClusterLayer(num_classes=10, class_weight=class_weights, name='cluster')
cluster = scl([x, labels])
model = tf.keras.Model(inputs=[input, labels], outputs=[output, cluster])


# -----------------------  train --------------------------------
# create fake clusters to pass as argument
train_cl = np.zeros((train_images.shape[0],))
test_cl = np.zeros((test_images.shape[0],))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
              loss={'output': SparseWeightedSoftmaxLoss2(10, class_weights, from_logits=True),
                    'cluster': WeightedClusterLoss(class_weights, _lambda=0.0)},
              metrics={'output': [tf.keras.metrics.SparseCategoricalAccuracy()]})
# model.train_on_batch([train_images[:32], train_labels[:32]], [train_labels[:32], train_cl[:32]])
model.fit([train_images, train_labels], [train_labels, train_cl],
          epochs=35,
          class_weight={'output': class_weights},
          validation_data=([test_images, test_labels], [test_labels, test_cl]))

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
#               loss={'output': SparseWeightedSoftmaxLoss(10, class_weights, from_logits=True)},
#               metrics={'output': ['accuracy']})
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels))

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
#               loss={'output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                     'cluster': WeightedClusterLoss(class_weights, _lambda=0.1)},
#               metrics={'output': ['accuracy']})
# model.fit([train_images, train_labels], [train_labels, train_cl], epochs=10,
#           validation_data=([test_images, test_labels], [test_labels, test_cl]))

