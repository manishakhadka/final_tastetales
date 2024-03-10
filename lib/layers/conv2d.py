import numpy as np
import tensorflow as tf

from lib.utils import conv_utils


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, strides=(1, 1), padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 2, 'kernel_size')
        self.activation = tf.keras.activations.get(activation)
        self.strides = strides
        self.padding = padding.upper()  # TF expects 'VALID' or 'SAME' in uppercase

        # Initializing kernel and bias
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # Kernel shape: (kernel_height, kernel_width, input_channels, output_channels)
        kernel_shape = (*self.kernel_size, input_shape[-1], self.filters)

        # Initializing the kernel and bias variables
        self.kernel = self.add_weight(
            name='kernel', shape=kernel_shape, initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(
            self.filters,), initializer='zeros', trainable=True)

        super().build(input_shape)

    def call(self, inputs):
        # Performing the convolution
        x = tf.nn.conv2d(inputs, self.kernel, strides=[
                         1, *self.strides, 1], padding=self.padding)

        # Adding the bias
        x = tf.nn.bias_add(x, self.bias)

        # Applying the activation function (if any)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.padding == 'SAME':
            out_height = np.ceil(
                float(input_shape[1]) / float(self.strides[0]))
            out_width = np.ceil(float(input_shape[2]) / float(self.strides[1]))
        else:  # VALID
            out_height = np.ceil(
                float(input_shape[1] - self.kernel_size[0] + 1) / float(self.strides[0]))
            out_width = np.ceil(
                float(input_shape[2] - self.kernel_size[1] + 1) / float(self.strides[1]))
        return (input_shape[0], int(out_height), int(out_width), self.filters)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'strides': self.strides,
            'padding': self.padding,
        })
        return config
