"""Keras networks

Most of these are pretty heavily copied from other places, and will be linked
when they are. The point of this repository (at time of writing) isn't to get
networks exactly right - it's to train on multiple GPUs, regardless of network.
"""

from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D


class VGG16():
    """VGG16 network

    Note that the number of filters in each layer has been halved to make the
    network small enough to fit on a GPU with 12G of memory.

    Code heavily taken from:
        https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
    """

    def __init__(self):
        """Init

        The point of this class is to house functionality to possibly build out
        networks later. It might be a little unncessary to house it in a class
        right now, but ¯\_(ツ)_/¯.
        """

        pass

    @staticmethod
    def _conv_block(layer, num_conv_layers, num_filters):
        """Build a conv block on top of inputs

        :param inputs: Keras Layer object representing the VGG net up to this
         point
        :param num_conv_layers: int for the number of convolutional layers to
         include in this block
        :param num_filters: int for the number of filters per convolutional
         layer
        """

        for _ in  range(num_conv_layers - 1):
            layer = Conv2D(
                filters=num_filters, kernel_size=(3, 3), padding='same',
                activation='relu'
            )(layer)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

        return layer

    def build(self, input_shape, num_classes):
        """Build the network

        :param input_shape: tuple holding the shape of the input
        """

        inputs = Input(shape=input_shape)

        layer = self._conv_block(
            layer=inputs, num_conv_layers=2, num_filters=32
        )
        layer = self._conv_block(
            layer=inputs, num_conv_layers=2, num_filters=64
        )
        layer = self._conv_block(
            layer=inputs, num_conv_layers=3, num_filters=128
        )
        layer = self._conv_block(
            layer=inputs, num_conv_layers=3, num_filters=256
        )

        layer = Flatten()(layer)
        layer = Dense(units=2048, activation='relu')(layer)
        layer = Dense(units=2048, activation='relu')(layer)
        outputs = Dense(units=num_classes, activation='softmax')(layer)

        return inputs, outputs
