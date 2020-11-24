import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer

"""
Weighted Softmax-Cluster loss: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7249188/

the referred Softmax loss relates to the CategoricalCrossEntropy loss

"""

# create Custom Cluster layer
class SparseClusterLayer(Layer):
    def __init__(self, num_classes, class_weight, alpha=1, gamma=1, **kwargs):
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.alpha = alpha
        self.gamma = gamma
        super(SparseClusterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # split input
        features = input_shape[0]

        # initialize cluster
        cluster_init = tf.constant_initializer(0)
        self.cluster = tf.Variable(name='clusters',
                                   initial_value=cluster_init(shape=(self.num_classes, features[-1]), dtype='float32'),
                                   trainable=False)

        super(SparseClusterLayer, self).build(input_shape)

    def call(self, x):
        # split input
        features = x[0]
        labels = x[1]
        features = tf.expand_dims(features, axis=1)

        # ------------------- compute Loss --------------------
        # compute numerator
        ci = tf.gather(self.cluster, labels, axis=0)
        nume = tf.pow(tf.norm(features - ci, axis=2), 2)

        # compute denominator
        cj = tf.reshape(self.cluster, [-1])  # flatten cj
        cj = tf.expand_dims(cj, axis=0)  # add extra axis
        cj = tf.repeat(tf.expand_dims(cj, axis=0), tf.shape(labels)[0], axis=0)  # expand and repeats
        ci_tile = tf.tile(ci, [1, 1, self.num_classes])
        denom = tf.pow(tf.norm(cj - ci_tile, axis=2), 2) + self.alpha

        loss = nume / denom

        # ------------------- Update cluster --------------------
        # cast y_true to one hot
        labels_one_hot = tf.cast(labels, dtype=tf.uint8)
        labels_one_hot = tf.one_hot(labels_one_hot, self.num_classes)
        labels_one_hot = tf.cast(labels_one_hot, dtype='float32')

        # compute numerator
        labels_tile = tf.repeat(tf.expand_dims(labels_one_hot, axis=3), tf.shape(features)[-1], axis=3)
        cluster = tf.expand_dims(self.cluster, axis=0)
        cj = tf.multiply(labels_tile, cluster)
        xj = tf.repeat(tf.expand_dims(features, axis=2), self.num_classes, axis=2)
        xj = tf.multiply(labels_tile, xj)
        nume = xj - cj

        # compute denominator
        denom = tf.repeat(tf.expand_dims(denom, axis=2), self.num_classes, axis=2)
        denom = tf.repeat(tf.expand_dims(denom, axis=3), tf.shape(features)[-1], axis=3)

        # compute delta center and apply weights
        delta_c = nume / denom
        cw = tf.expand_dims(self.class_weight, axis=0)  # add extra axis
        cw = tf.repeat(tf.expand_dims(cw, axis=0), tf.shape(labels)[0], axis=0)
        cw = tf.multiply(labels_one_hot, cw)
        cw = tf.repeat(tf.expand_dims(cw, axis=3), features.shape[1], axis=3)
        weighted_delta_c = tf.multiply(cw, delta_c)
        weighted_delta_c = tf.reduce_sum(weighted_delta_c, axis=0)

        # update clusters
        new_cluster = self.gamma * weighted_delta_c
        new_cluster = tf.reshape(new_cluster, shape=tf.shape(self.cluster))
        self.cluster.assign_sub(new_cluster)

        # return loss
        return loss


class SparseWeightedSoftmaxLoss(Loss):
    """
    Expect labels to be provided as integers.
    """

    def __init__(self, num_classes, class_weights, from_logits=False):
        super().__init__()
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # print("[call] shape y_true", y_true.shape)
        # cast y_true to one hot
        y_true = tf.cast(y_true, dtype=tf.uint8)
        y_true = tf.one_hot(y_true, self.num_classes)
        y_true = tf.cast(y_true, dtype='float32')
        y_true = tf.squeeze(y_true)
        # print("shape y_true_one_hot", y_true.shape)

        # get batch size
        batch_size = tf.shape(y_pred)[0]
        batch_size = tf.cast(batch_size, dtype=y_pred.dtype)

        # get softmax loss
        if self.from_logits:
            loss = self._softmax_loss_with_logits(y_true, y_pred)
        else:
            loss = self._softmax_loss(y_true, y_pred)

        # apply class weights and compute the sum
        cw = tf.repeat(tf.expand_dims(self.class_weights, axis=0), tf.shape(y_pred)[0], axis=0)
        return -tf.reduce_sum(tf.multiply(cw, loss)) / batch_size

    def _softmax_loss_with_logits(self, y_true, y_pred):
        """ the function is equal to the CategoricalCrossEntropy """
        # compute a to apply log sum exp trick
        a = tf.reduce_max(y_pred, axis=1)
        a = tf.repeat(tf.expand_dims(a, axis=1), self.num_classes, axis=1)

        sum_log = tf.math.log(tf.reduce_sum(tf.exp(y_pred - a), axis=1))
        sum_log = a + tf.repeat(tf.expand_dims(sum_log, axis=1), self.num_classes, axis=1)
        return tf.multiply(y_true, y_pred - sum_log)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.math.maximum(1e-12, y_pred)
        return tf.multiply(y_true, tf.math.log(y_pred))


class WeightedClusterLoss(Loss):
    def __init__(self, class_weights, _lambda=1):
        super().__init__()
        self.class_weights = class_weights
        self._lambda = _lambda

    def call(self, y_true, y_pred):
        # apply class weights and compute the sum
        cw = tf.repeat(tf.expand_dims(self.class_weights, axis=0), tf.shape(y_pred)[0], axis=0)
        return .5 * self._lambda * tf.reduce_sum(tf.multiply(cw, y_pred))


if __name__ == '__main__':
    # ************************************   TEST  Layer  *******************************
    x = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]])
    cw = tf.convert_to_tensor([.5, .5, 2])
    y_true = tf.convert_to_tensor([1, 2])
    y_true = np.expand_dims(y_true, axis=1)
    print("shape y_true", np.shape(y_true))
    x_shape = tf.TensorShape((None, 4))
    label_shape = tf.TensorShape((None, 3))
    input_shape = [x_shape, label_shape]
    cl_layer = SparseClusterLayer(num_classes=3, class_weight=cw, name='clusterlosslayer')
    cl_layer.build(input_shape)
    res = cl_layer([x, y_true])
    print("res", res.shape)
    print(res)
    print()

    # ************************************   TEST LOSS   *******************************
    y_true = tf.convert_to_tensor([1, 2])
    y_true = tf.expand_dims(y_true, axis=1)
    print("[declare] shape y_true", y_true.shape)
    y_pred = tf.convert_to_tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    centers = tf.convert_to_tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    x = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]])
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = scce(y_true, y_pred).numpy()
    print("SparseCategoricalCrossentropy", loss)
    # return 1.177
    scce = SparseWeightedSoftmaxLoss(3, [1., 1., 1.], from_logits=True)
    loss = scce(y_true, y_pred).numpy()
    print("SparseWeightedSoftmaxLoss", loss)

    scce = WeightedClusterLoss([1., 1., 1.])
    loss = scce(y_true, y_pred).numpy()
    print("WeightedClusterLoss", loss)
    print()