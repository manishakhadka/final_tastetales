import numpy as np
import tensorflow as tf

from lib.activations import elu, softmax
from lib.initializers import glorot_init, he_init


class Dense(tf.keras.layers.Layer):
    '''For layers, we need to override the tf.keras.layers.Layer.'''

    def __init__(self, units, activation=None, initializer=None, **kwargs):
        'Define those attributes we would be using inside other functions.'

        # units is the number of units in the layer
        self.units = units

        # super function to handle those values we might have missed
        # "Take on the default behaviour, unless explictly over rided"
        super().__init__(**kwargs)

        if not activation:
            self.activation = elu()

        # parse if activation is given in string format
        if isinstance(activation, str):
            self.activation = tf.keras.activations.get(activation)

        # specifying default initializer
        if not initializer:
            self.initializer = he_init
        else:
            self.initializer = initializer

    def build(self, batch_input_shape):
        '''Main role of the build() is to create the layers variables -> weights and biases.
        It is called the first time the layer is built. '''

        self.kernel = self.add_weight(
            name='kernel',
            shape=[batch_input_shape[-1], self.units],
            initializer=self.initializer)

        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros')

        # we call super() at the only the end this is
        # to let keras know that the layer has been built
        super().build(batch_input_shape)

    def call(self, x):
        '''Performs what the layer is supposed to do 
        -> Matrix multiplication of input X with the kernel 
        -> Add the above output with bias term.'''

        # @ -> matrix multiplication (overridden for tensors by default)
        return self.activation(x @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        '''Simply returns the shape of the tensors output, which is same as input
        except the final dimension that is replaced with number of units in the layer.

        Output must be of TensorShape datatype.

        Note: batch_input_shape is of type `tf.TensorShape` which can be converted 
        to list with function as_list().
        '''

        return tf.TensorShape(batch_input_shape.as_list()[:-1], [self.units])

    def get_config(self):
        'Override to be able to save the custom object later on.'

        parent_configs = super().get_config()
        return {
            **parent_configs,
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "initializer": tf.keras.initializers.serialize(self.initializer),
        }
