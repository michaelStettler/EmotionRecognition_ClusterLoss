import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pandas as pd
import tensorflow_datasets as tfds


num_data = 512
batch_size = 32
num_epochs = 5

# create image folder
img_path = 'train_img'
if not os.path.exists(img_path):
    os.makedirs(img_path)

# create small subset of dataset to build a csv with the image path and label
df = pd.DataFrame(columns=['img_path', 'label'])
ds = tfds.load('cifar10', split='train', as_supervised=True)
ds = ds.take(num_data)
idx = 0
for image, label in tfds.as_numpy(ds):  # example is (image, label)
    img_name = os.path.join(img_path, 'img_'+str(idx)+'.jpg')
    # create csv
    df = df.append({'img_path': img_name, 'label': str(label)}, ignore_index=True)
    # save images
    cv2.imwrite(img_name, image)

    idx += 1


# define a simple Center layer
class CenterLayer(Layer):
    """
    Toy example of my center Layer
    """
    def __init__(self, num_classes, **kwargs):
        super(CenterLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        # split input
        features = input_shape[0]
        # initialize cluster
        cluster_init = tf.constant_initializer(0)
        self.centers = tf.Variable(name='centers',
                                   initial_value=cluster_init(shape=(self.num_classes, features[-1]), dtype='float32'),
                                   trainable=False)

    def call(self, inputs):
        image = inputs[0]
        labels = inputs[1]
        # compute center loss
        # ...
        # update centers
        # ...

        # should return center loss
        # print("shape x", tf.shape(x))
        # print("shape labels", tf.shape(labels))
        return labels



# create a simple model with two input and two outputs to mimic the center loss architecture
input = tf.keras.Input(shape=(32, 32, 3))
label = tf.keras.Input(shape=(10, ))
x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(10, name='output')(x)
center = CenterLayer(10, name='center')([x, label])
model = tf.keras.Model(inputs=[input, label], outputs=[output, center])

model.compile(loss={'output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    'center': tf.keras.losses.CategoricalCrossentropy(from_logits=True)},  # loss should also be changed to a CenterLoss function
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics={'output': ['accuracy']})
print(model.summary())

# create generator
generator = tf.keras.preprocessing.image.ImageDataGenerator(vertical_flip=True)
train_gen = generator.flow_from_dataframe(df, x_col='img_path', y_col='label',
                                          class_mode='categorical',
                                          target_size=(32, 32),
                                          batch_size=batch_size)


def my_generator(generator):
    while True:
        batch = generator.next()
        img = batch[0]
        labels = batch[1]
        labels_size = np.shape(labels)
        cluster = np.zeros(labels_size)
        x = [img, labels]
        y = [labels, cluster]

        yield x, y


train_generator = my_generator(train_gen)

# fit model
model.fit(train_generator,
          epochs=num_epochs,
          steps_per_epoch=num_data/batch_size,
          workers=4,
          use_multiprocessing=True)
