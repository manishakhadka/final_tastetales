import tensorflow as tf


class MaxPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding.upper()

    def call(self, inputs):
        return tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config
