import numpy as np
import tensorflow as tf

from src.model_functions.WeightedSoftmaxCluster import ClusterLayer
from src.model_functions.WeightedSoftmaxCluster import WeightedSoftmaxLoss2
from src.model_functions.WeightedSoftmaxCluster import WeightedClusterLoss

"""
run:  python3 -m src.model_functions.test_clustLoss_mnist

"""

# load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_train = x_train / 255.0
x_test = np.expand_dims(x_test, axis=3)
x_test = x_test / 255.0
test_labels = y_test
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print("mnist")
print("shape x_train", np.shape(x_train))
print("min max x_train", np.amin(x_train), np.amax(x_train))
print("shape y_train", np.shape(y_train))

# construct model
class_weights = np.ones(10).astype('float32')

input = tf.keras.Input(shape=(28, 28, 1))
label = tf.keras.Input(shape=(10, ))
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input)
x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(2, name='embedding')(x)
x = tf.keras.layers.PReLU()(x)
cluster = ClusterLayer(10, class_weights, name='ClusterLayer')([x, label])
output = tf.keras.layers.Dense(10, name='output')(x)


model = tf.keras.Model(inputs=[input, label], outputs=[output, cluster])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9),
              loss={'output': WeightedSoftmaxLoss2(10, class_weights, from_logits=True),
                    'ClusterLayer': WeightedClusterLoss(class_weights)},
              metrics={'output': [tf.keras.metrics.CategoricalAccuracy()]},
              loss_weights=[1, .5])
print(model.summary())

# fit
# create fake clusters to pass as argument
train_cl = np.zeros((x_train.shape[0],))
test_cl = np.zeros((x_test.shape[0],))

print("shape x_train", np.shape(x_train))
print("shape y_train", np.shape(y_train))
print("shape train_cl", np.shape(train_cl))

# model.train_on_batch([x_train[:32], y_train[:32]], y=[[y_train[:32], train_cl[:32]]])
hist = model.fit([x_train, y_train], y=[y_train, train_cl], epochs=180, batch_size=256,
          validation_data=([x_test, y_test], [y_test, test_cl]))


# evaluate embedding
emb_model = tf.keras.Model(inputs=model.input[0], outputs=model.get_layer('embedding').output)
preds = emb_model.predict(x_test)
print("shape preds", np.shape(preds))

print("shape test_labels", np.shape(test_labels))
print("test_labels_sample[:10]")
print(test_labels[:10])

# plot
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']

for i in range(10):
    plt.plot(preds[test_labels == i, 0].flatten(), preds[test_labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.savefig('cluster.png')

# plot history
# summarize history for accuracy
plt.figure()
plt.plot(hist.history['output_categorical_accuracy'])
plt.plot(hist.history['val_output_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['output_loss'])
plt.plot(hist.history['ClusterLayer_loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'train_output_loss', 'train_clust_loss', 'test'], loc='upper left')
plt.savefig('loss.png')

