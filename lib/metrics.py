import tensorflow as tf
import numpy as np

# Sparse Categorical Accuracy
class scacc_metric(tf.keras.metrics.Metric):
    '''To create a metric, we need to override the tf.keras.metrics.Metric class.
    Here we are creating a stateful metric, which keeps updating itself with streaming scores.
    No new ARGS to init, hence get_config is not needed here.
    
    Note: Even though we create new varibles in it, we don't require saving metrics. 
    If you ever need to save the metric, say a special value which the metric uses, 
    then simply create a get_config() and override it.       
    '''
    
    def __init__(self, **kwargs):
        '''Init as before, notice we have no new arguments we are explictly 
        overriding or using. It is implictly passed as **kwargs to be handled by super.
        '''
        super().__init__(**kwargs)
        
        # intialize total & count to calculate the mean when required
        self.total = self.add_weight("total", initializer='zeros')
        self.count = self.add_weight("count", initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weights=None):
        '''Gets called inside the loop for EVERY STEP OF EPOCH. We compute the metric
        & update the total values so that later on we can utilize them. 
        
        This makes more sense for metrics such as precision.'''
        
        # calculate Sparse Categrical accuracy
        metrics = self.calc_SCAcc(y_true, y_pred)
        
        # update the total scores along with count for later purpose
        self.total.assign_add(tf.reduce_sum(metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))
    
    def calc_SCAcc(self, y_true, y_pred):
        '''Helper function to compute the metrics given a batch.'''
        
        scores = tf.cast(tf.equal(
            tf.cast(tf.reduce_max(y_true, axis=-1), dtype=tf.float32), 
            tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32),
        ), dtype=tf.float32)
        
        return scores
    
    def result(self):
        '''Gets called at the END of each epoch, write the function accordingly. 
        Here, it calculates the mean.'''
        
        # I am fully aware that we could use a simple tf.reduce_mean(metrics) :)
        return self.total / self.count