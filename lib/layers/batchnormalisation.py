import tensorflow as tf


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        # Moving mean and variance are not trainable parameters.
        self.moving_mean = None
        self.moving_variance = None
        # Gamma (scale) and beta (offset) are trainable parameters.
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        # Initialize gamma and beta, with shape of the last dimension of the input.
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(
            name='gamma', shape=shape, initializer='ones', trainable=True)
        self.beta = self.add_weight(
            name='beta', shape=shape, initializer='zeros', trainable=True)

        # Moving mean and variance are not trainable. They are updated during training.
        self.moving_mean = self.add_weight(
            name='moving_mean', shape=shape, initializer='zeros', trainable=False)
        self.moving_variance = self.add_weight(
            name='moving_variance', shape=shape, initializer='ones', trainable=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Compute the mean and variance of the current batch.
            batch_mean, batch_variance = tf.nn.moments(
                inputs, axes=list(range(len(inputs.shape) - 1)))

            # Update the moving mean and variance.
            self.moving_mean.assign(
                self.moving_mean * self.momentum + batch_mean * (1 - self.momentum))
            self.moving_variance.assign(
                self.moving_variance * self.momentum + batch_variance * (1 - self.momentum))

            mean, variance = batch_mean, batch_variance
        else:
            # Use the moving averages during inference.
            mean, variance = self.moving_mean, self.moving_variance

        # Normalize the inputs.
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        # Scale and offset
        return self.gamma * normalized_inputs + self.beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon': self.epsilon,
            'momentum': self.momentum,
        })
        return config
