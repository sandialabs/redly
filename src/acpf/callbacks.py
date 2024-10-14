#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam


class TimeCallback(Callback):
    '''
    Callback to report on training times
    '''   
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.starttime)


class ESCallback(Callback):
    '''
    Callback to handle early stopping
    '''
    def __init__(self, patience=20, valname='val_bmse_metric'):
        super(ESCallback, self).__init__()
        self.patience = patience
        self.valname = valname
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.valname)
        if np.less(current, self.best): 
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience: # ran out of patience, stop training
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        print("Restoring model weights from the end of the best epoch")
        self.model.set_weights(self.best_weights) # restore best_weights
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        
        
class LDCallback(Callback):
    '''
    Callback to implement Lagrangian Dual (LD) training
    (Fioretto, Mak, and Hentenryck. Combining Deep Learning and 
     Lagrangian Dual Methods, 2020)
    Optionally resets optimizer state after each LD update step
    '''
    def __init__(self, opt, rho=1e-2, ld_int=10, ld_prog=0, 
                 steps_per_execution=1, reset=False):
        super(LDCallback, self).__init__()
        self.opt = opt # TODO currently assumes Adam optimizer
        self.rho = rho # LD rho update parameter
        self.ld_int = ld_int # initial LD interval length
        self.ld_prog = ld_prog # LD arithmetic progression parameter
        self.reset = reset
        
    def on_train_begin(self, logs=None):
        # initialize step count
        self.ld_count = 0
        
        if self.reset:
            # save optimizer state variables for resetting
            self.init_iterations = tf.Variable(K.get_value(self.opt.iterations), 
                                               dtype=self.opt.iterations.dtype, trainable=False)
            self.init_lr = tf.Variable(K.get_value(self.opt.lr), dtype=self.opt.lr.dtype,
                                       trainable=False)
            self.init_beta_1 = tf.Variable(K.get_value(self.opt.beta_1),
                                           dtype=self.opt.beta_1.dtype, trainable=False)
            self.init_beta_2 = tf.Variable(K.get_value(self.opt.beta_2),
                                           dtype=self.opt.beta_2.dtype, trainable=False)
        
    def on_epoch_end(self, epoch, logs=None):
        self.ld_count += 1
        if (self.ld_count == self.ld_int) and not (self.model.stop_training): # LD update step
            print("Updating Lagrangian Dual parameters")
            for layer in self.model.layers:
                if 'conname' in dir(layer): # update each eta per rho + violation
                    K.set_value(layer.eta, K.get_value(layer.eta) + self.rho*logs.get(layer.conname))
            
            if self.reset:
                # reset optimizer state
                K.set_value(self.opt.iterations, K.get_value(self.init_iterations))
                K.set_value(self.opt.lr, K.get_value(self.init_lr))
                K.set_value(self.opt.beta_1, K.get_value(self.init_beta_1))
                K.set_value(self.opt.beta_2, K.get_value(self.init_beta_2))
            
            # reset ld_count and increment ld_int
            self.ld_count = 0
            self.ld_int += self.ld_prog            
            
            
class LTHCallback(Callback):
    '''
    Callback to implement Lottery Ticket Hypothesis (LTH) pruning
    (Frankle and Carbin 2018)
    '''
    def __init__(self, ithresh=.2, othresh=.1, mask_data={}, lname='mdense'):
        super(LTHCallback, self).__init__()
        self.ithresh = ithresh # internal weight threshold
        self.othresh = othresh # output layer weight threshold
        self.mask_data = mask_data # mask/init_weights dictionary
        self.lname = lname
        
    def on_train_begin(self, logs=None):
        mask_data = {}
        for layer in self.model.layers:
            if layer.name.startswith('%s_'%self.lname):
                # If there is no prior mask data, snapshot the pretrained weights
                if not layer.name in self.mask_data:
                    weights = layer.get_weights()[0]
                    mask = np.ones(weights.shape, dtype=np.float32)
                    mask_data[layer.name] = {'mask':mask, 'weights':weights}
        self.mask_data.update(mask_data)

    def on_train_end(self, logs=None):
        mask_data = {}
        for layer in self.model.layers:
            if layer.name.startswith('%s_'%self.lname):
                # get trained weights and original mask
                weights = layer.get_weights()[0]
                mask = self.mask_data[layer.name]['mask'].reshape(-1)
                # sort nonzero indices and count z's
                ind = np.argsort(np.abs(mask*weights.reshape(-1)))
                nz = len(np.where(mask==0)[0])
                # select appropriate threshold
                thresh = self.ithresh if str.isdigit(layer.name.split('_')[1]) \
                    else self.othresh
                # mask out bottom thresh percentile of remaining nz's
                mask[ind[nz:nz+int(thresh*(len(ind)-nz))]] = 0
                mask = mask.reshape(weights.shape)
                # update mask dictionary
                mask_data[layer.name] = {'mask':mask, 
                                         'weights':self.mask_data[layer.name]['weights']}
        self.mask_data.update(mask_data)