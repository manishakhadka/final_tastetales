import tensorflow as tf
import numpy as np


# For softmax layers
def glorot_init(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


# For elu layers
def he_init(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / shape[0])
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
