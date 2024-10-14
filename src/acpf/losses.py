#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric


class BMSEMetric(Metric):
    '''
    MSE loss that incorporates boundary value flag
    L(y,\hat{y}) only contributes if boundary value flag is set
    Written as a metric to separate from regularization losses
    '''
    def __init__(self, model_data, name='bmse'):
        super(BMSEMetric, self).__init__()
        self.state = self.add_weight(name=name, initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name=name+'_cnt', initializer='zeros', dtype=tf.float32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        b = tf.gather(y_true, y_true.shape[-1]-1, axis=1)
        y_true = tf.gather(y_true, np.arange(y_true.shape[-1]-1), axis=1) # remove boundary flag
        y_pred = tf.gather(y_pred, np.arange(y_true.shape[-1]), axis=1) # remove flow vars
        out = tf.reduce_sum(b*tf.reduce_sum(tf.square(y_true-y_pred), axis=1))
        self.state.assign_add(out)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
    def result(self):
        return tf.math.divide_no_nan(self.state, self.count)

    def reset_states(self):
        self.state.assign(0)
        self.count.assign(0)

        
class BMSELoss(Loss):
    '''
    MSE loss that incorporates boundary value flag
    L(y,\hat{y}) only contributes if boundary value flag is set
    '''   
    def __init__(self, model_data):
        super(BMSELoss, self).__init__()

    def call(self, y_true, y_pred):
        b = tf.gather(y_true, y_true.shape[-1]-1, axis=1)
        y_true = tf.gather(y_true, np.arange(y_true.shape[-1]-1), axis=1) # remove boundary flag
        y_pred = tf.gather(y_pred, np.arange(y_true.shape[-1]), axis=1) # remove flow vars
        return b*tf.reduce_mean(tf.square(y_true-y_pred), axis=1)

    
class BinaryMSEMetric(Metric):
    '''
    Combines Binary + MSE metric, with optional weights
    Assumes output format y=[y_sec] + \sum_{c\in C} [y_c] + [b_flag]
    L(y,\hat{y}) = b_flag*w_sec*L(y_sec, \hat{y}_sec) + \sum_{c\in C}b_flag*w_c*L(y_c, \hat{y}_c)
    Written as a metric to separate from regularization losses
    '''
    def __init__(self, model_data, w_sec=1, w_c=None, name='binmse'):
        super(BinaryMSEMetric, self).__init__()
        self.contingencies = [con for con in model_data['contingencies'] 
                              if not con==model_data['nominal']]
        inds = model_data['inds']
        self.inds = {}
        for con in self.contingencies:
            # need +1 to account for security index
            self.inds[con] = 1+np.concatenate([inds[0][con], inds[3][con], inds[4][con], inds[5][con], 
                                               inds[6][con], inds[7][con], inds[8][con]])
        self.w_sec = w_sec
        if w_c is not None:
            self.w_c = w_c
        else:
            # normalize regression weights by num. contingencies
            w = 1/float(len(self.contingencies))
            self.w_c = {con:1/w for con in self.contingencies}
        self.state = self.add_weight(name=name, initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name=name+'_cnt', initializer='zeros', dtype=tf.float32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        b = tf.gather(y_true, y_true.shape[-1]-1, axis=1)
        y_sec_true = tf.gather(y_true, 0, axis=1)
        y_sec_pred = tf.gather(y_pred, 0, axis=1)
        out = b*self.w_sec*K.binary_crossentropy(y_sec_true, y_sec_pred, from_logits=False)
        for con in self.contingencies:
            y_c_true = tf.gather(y_true, self.inds[con], axis=1)
            y_c_pred = tf.gather(y_pred, self.inds[con], axis=1)
            out += b*self.w_c[con]*tf.reduce_mean(tf.square(y_c_true-y_c_pred), axis=1)
        self.state.assign_add(tf.reduce_sum(out))
        self.count.assign_add(tf.cast(tf.size(y_pred), tf.float32))
        
    def result(self):
        return tf.math.divide_no_nan(self.state, self.count)

    def reset_states(self):
        self.state.assign(0)
        self.count.assign(0)

    
class BinaryMSELoss(Loss):
    '''
    Combines Binary + MSE loss, with optional weights
    Assumes output format y=[y_sec] + \sum_{c\in C} [y_c] + [b_flag]
    L(y,\hat{y}) = b_flag*w_sec*L(y_sec, \hat{y}_sec) + \sum_{c\in C}b_flag*w_c*L(y_c, \hat{y}_c)
    '''
    def __init__(self, model_data, w_sec=1, w_c=None):
        super(BinaryMSELoss, self).__init__()
        self.contingencies = [con for con in model_data['contingencies'] 
                              if not con==model_data['nominal']]
        inds = model_data['inds']
        self.inds = {}
        for con in self.contingencies:
            # need +1 to account for security index
            self.inds[con] = 1+np.concatenate([inds[0][con], inds[3][con], inds[4][con], inds[5][con], 
                                               inds[6][con], inds[7][con], inds[8][con]])
        self.w_sec = w_sec
        if w_c is not None:
            self.w_c = w_c
        else:
            # normalize regression weights by num. contingencies
            w = 1/float(len(self.contingencies))
            self.w_c = {con:1/w for con in self.contingencies}
        
    def call(self, y_true, y_pred):
        b = tf.gather(y_true, y_true.shape[-1]-1, axis=1)
        y_sec_true = tf.gather(y_true, 0, axis=1)
        y_sec_pred = tf.gather(y_pred, 0, axis=1)
        loss = b*self.w_sec*K.binary_crossentropy(y_sec_true, y_sec_pred, from_logits=False)
        for con in self.contingencies:
            y_c_true = tf.gather(y_true, self.inds[con], axis=1)
            y_c_pred = tf.gather(y_pred, self.inds[con], axis=1)
            loss += b*self.w_c[con]*tf.reduce_mean(tf.square(y_c_true-y_c_pred), axis=1)
        return loss