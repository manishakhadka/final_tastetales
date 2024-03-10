# import tensorflow as tf
# import numpy as np


# class elu(tf.keras.layers.Layer):
#     def __init__(self, alpha=1., **kwargs):
#         '''As always contains the super class init. Or any new attribute defined comes here.
#         This is our custom implementation of exponential linear unit function.'''

#         super().__init__(**kwargs)
#         self.alpha = alpha

#     def call(self, z):
#         'Call defines the behaviour of the layer.'

#         z = tf.cast(z, dtype=tf.float32)
#         return tf.where(z > 0, z, self.alpha * (tf.exp(z) - 1))

#     def get_config(self):
#         '''Required only when we need to save config of our custom model &
#         when we have defined new attributes in init that needs saving'''

#         return {**super().get_config(), "alpha": self.alpha}


# def softmax(z, axis=-1):
#     '''As stated before, we dont have to over ride classes all the time.
#     Sometimes a simple python function written with tf functions would suffice.'''

#     # calculate the number of dimensions
#     # ndim = len(tf.shape(z))
#     ndim = z.ndim

#     # we compute only with dimensions are >= 2, throw error otherwise
#     if ndim >= 2:
#         e = tf.exp(z - tf.reduce_max(z, axis=axis, keepdims=True))
#         s = tf.reduce_sum(e, axis=axis, keepdims=True)
#         return e / s

#     else:
#         raise ValueError('Cannot apply softmax to a tensor that is 1D. '
#                          'Received input: %s' % z)


# class modrelu(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(modrelu, self).__init__(**kwargs)

#     # provide input_shape argument in the build method
#     def build(self, input_shape):
#         # You should pass shape for your variable
#         self.b = K.variable(value=np.random.rand(*input_shape)-0.5,
#                             dtype='float32')
#         # Be sure to call this at the end
#         super(modrelu, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         assert inputs.dtype == tf.complex64
#         ip_r = tf.math.real(inputs)
#         ip_i = tf.math.imag(inputs)
#         comp = tf.complex(ip_r, ip_i)
#         ABS = tf.math.abs(comp)
#         ANG = tf.math.angle(comp)
#         ABS = K.relu(self.b + ABS)
#         op_r = ABS * K.sin(ANG)  # K.dot ??
#         op_i = ABS * K.cos(ANG)
#         # return single tensor in the call method
#         return tf.complex(op_r, op_i)


# class relu(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(relu, self).__init__(**kwargs)

#     def call(self, inputs, **kwargs):
#         return tf.maximum(inputs, 0)


# class sigmoid(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(sigmoid, self).__init__(**kwargs)

#     def call(self, inputs, **kwargs):
#         return 1 / (1 + tf.exp(-inputs))


# fallback to the default
from keras.activations import elu, softmax, sigmoid, relu
