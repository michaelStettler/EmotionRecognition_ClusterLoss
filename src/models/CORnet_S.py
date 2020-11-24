import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import Model


class CORblock_S:
    scale = 4  # Scale of the bottleneck convolution channels
    chanDim = -1  # If format is 'channels last'

    def __init__(self, out_channels, layer_name, times=1):
        # super().__init__()
        self.out_channels = out_channels
        self.layer_name = layer_name
        self.times = times

        # declare the shared layers
        self.conv_input = Conv2D(self.out_channels, (1, 1),
                                 padding='SAME',
                                 name=layer_name + '_convInp')
        self.skip = Conv2D(self.out_channels, (1, 1),
                           strides=2,
                           padding='SAME',
                           use_bias=False,
                           name=layer_name + '_skip')
        self.conv1 = Conv2D(self.out_channels * self.scale, (1, 1),
                            padding='SAME',
                            use_bias=False,
                            name=layer_name + '_conv1')
        self.conv2 = Conv2D(self.out_channels * self.scale, (3, 3),
                            padding='SAME',
                            strides=1,
                            use_bias=False,
                            name=layer_name + '_conv2')
        self.conv3 = Conv2D(self.out_channels, (1, 1),
                            padding='SAME',
                            use_bias=False,
                            name=layer_name + '_conv3')

    # @staticmethod
    def CORblock_S(self, input):

        x = self.conv_input(input)

        for t in range(self.times):

            if t == 0:
                skip = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name + '_BNSkip')(self.skip(x))
            else:
                skip = x

            x = self.conv1(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name + '_' + str(t) + '_BN1')(x)
            x = tf.keras.layers.Activation("relu", name=self.layer_name + '_' + str(t) + '_A1')(x)

            x = self.conv2(x)
            if t == 0:  # adding a stride=2 for t=0
                x = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name + '_' + str(t) + '_BN2')(x)
            x = tf.keras.layers.Activation("relu", name=self.layer_name + '_' + str(t) + '_A2')(x)

            x = self.conv3(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name + '_' + str(t) + '_BN3')(x)

            x += skip
            x = tf.keras.layers.Activation("relu", name=self.layer_name + '_' + str(t) + '_AOutput')(x)

        return x


def CORnet_S(classes=1000, from_logits=False):
    """
    Implementation of the CorNet_S architecture from:
    "CORnet: Modeling the Neural Mechanisms of Core Object Recognition"
    by Jonas Kubilius, Martin Schrimpf, Aran Nayebi, Daniel Bear, Daniel L. K. Yamins and James J. DiCarlo

    """

    # Define & implement model
    # Functional API format
    inputs = Input(shape=(224, 224, 3))

    # V1
    x = Conv2D(64, (7, 7), strides=2, padding='SAME', use_bias=False)(inputs)  # Check
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='SAME')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    # weight sharing for V2-V4-IT
    x = CORblock_S(128, layer_name="V2", times=2).CORblock_S(x)
    x = CORblock_S(256, layer_name="V4", times=4).CORblock_S(x)
    x = CORblock_S(512, layer_name="IT", times=2).CORblock_S(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(classes)(x)

    if not from_logits:
        x = Activation('softmax')(x)

    return Model(inputs, x, name='CORnetS')
