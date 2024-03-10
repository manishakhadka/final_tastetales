import tensorflow as tf
import numpy as np


class Concatenate(tf.keras.layers.Layer):
    def call(self, inputs):
        '''Note that inputs no matter how many come in through the 
        variable inputs, we need to do some tuple unpacking ourselves'''
        
        a, b = inputs
        return tf.concat([a, b], axis=1)
    
    def compute_output_shape(self, batch_input_shapes):
        'Output shape must be in TensorShape format.'
        return tf.TensorShape(
            batch_input_shapes[0][:-1],
            [batch_input_shapes[0][-1] + batch_input_shapes[1][-1]])