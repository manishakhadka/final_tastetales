import numpy as np
import tensorflow as tf


class Dropout(tf.keras.layers.Layer):
    def __init__(self, drate, **kwargs):
        # drate denotes the dropout rate probability
        self.drate = drate
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            # Use tf.shape(inputs) to get the dynamic shape
            mask = tf.random.uniform(shape=tf.shape(inputs)) > self.drate
            return tf.where(mask, inputs, 0) / (1 - self.drate)
        else:
            return inputs

    def get_config(self):
        # Override to be able to save the dropout rates
        return {**super().get_config(), "drate": self.drate}
