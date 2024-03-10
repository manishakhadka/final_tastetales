import numpy as np
import tensorflow as tf


class Flatten(tf.keras.layers.Layer):
    def call(self, x):
        'Simply reshape the input to required shape (Flatten)'
        return tf.reshape(x, shape=[x.shape[0], -1])

    def compute_output_shape(self, batch_input_shape):
        'Output shape must be in TensorShape format.'
        return tf.TensorShape([batch_input_shape[0]], [tf.reduce_prod(batch_input_shape[1:])])
