import numpy as np
import tensorflow as tf


class cross_entropy(tf.keras.losses.Loss):
    '''Cross entropy loss, loss layers inherit from keras.losses.loss class.
    Requires one hot encoded targets (categorical cross entropy)'''

    def __init__(self, epsilon=1e-12, **kwargs):
        '''We would see super().__init__(**kwargs) in almost every custom 
        object we would be building we do this, so that tf.keras takes care 
        of handling methods & attributes we may have failed to overload ourselves 
        '''
        super().__init__(**kwargs)

        # we use epsilon, a tiny miniscule value
        # to prevent the loss from becoming NAN
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        '''The main function of loss class that determines what the layer is supposed to do.
        Remember to use only `tf.keras.Backend` functions or tf functions. Others would slow 
        down the model terribly since internally all layers are converted to graphs. '''

        pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        N = pred.shape[0]
        ce = -tf.reduce_sum(tf.math.log(pred) * y_true) / N

        return ce

    def get_config(self):
        '''This is a must if we hope to save our model someday. 
        tf.keras.models.Model.save() requires this. Any new values we 
        had used that needs saving comes here. We over ride the super class
        config method and add our custom objects here.'''

        return {**super().get_config(), "epsilon": self.epsilon}


# Sparse cross entropy loss, loss layers inherit from keras.losses.loss class.
class s_cross_entropy(tf.keras.losses.Loss):
    '''Same as above but sparse_crossentropy loss.
    Takes in sparse targets (spare_categorical_crossentropy).'''

    def __init__(self, epsilon=1e-12, **kwargs):
        'Same as above, override parent class, define new variables here'
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        'Function that determines the behaviour of the layer'

        # one hot encode the targets and squeeze any dimensions of value 1
        y_true = tf.squeeze(tf.one_hot(y_true, depth=y_pred.shape[-1]))

        # exactly copied from above loss snippet
        pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        N = pred.shape[0]

        # we add the epsilon value since the model loss
        # hits NAN when trained for longer periods
        ce = -tf.reduce_sum(tf.math.log(pred + 1e-9) * y_true) / N

        return ce

    def get_config(self):
        'Override to save any custom objects we had made'
        return {**super().get_config(), "epsilon": self.epsilon}
