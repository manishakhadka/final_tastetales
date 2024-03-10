import tensorflow as tf

class GlobalMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=[1, 2])

    def get_config(self):
        return super().get_config().copy()
