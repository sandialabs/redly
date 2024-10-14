#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import tensorflow as tf
from tensorflow.keras.layers import Layer, ReLU
from tensorflow.keras.metrics import Mean

        
class FlowLayer(Layer):
    '''
    Computes violation between predicted and ground-truth flows 
    Predicted flows:  [v_pf, v_pt, v_qf, v_qt]
    Ground truth flows:  [pf, pt, qf, qt]
    '''
    def __init__(self, eta=0, name='flow'):
        super(FlowLayer, self).__init__()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        v_pf, v_pt, v_qf, v_qt, pf, pt, qf, qt = inputs
        v_f = tf.reduce_mean(tf.abs(tf.concat([v_pf-pf, v_pt-pt, v_qf-qf, v_qt-qt], axis=1)))
        self.add_metric(self.mean(v_f)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_f)
        
        return inputs


class KCLLayer(Layer):
    '''
    Computes KCL violations from the provided state variables and grid coefficient matrices
    Recall that KCL violations are linear in WCS formulation
    State vars:  [pg, pl, ql, wi, pr, qg, wo, v_pf, v_pt, v_qf, v_qt]
    Coefficient matrices:  K_p, K_q
    '''
    def __init__(self, K_p, K_q, eta=0, name='kcl'):
        super(KCLLayer, self).__init__()
        self.K_p = tf.convert_to_tensor(K_p)
        self.K_q = tf.convert_to_tensor(K_q)
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        pg, pl, ql, wi, pr, qg, wo, v_pf, v_pt, v_qf, v_qt = inputs
        v_p = tf.matmul(tf.concat([pg, pl, pr, v_pf, v_pt, wi, wo], axis=1), self.K_p)
        v_q = tf.matmul(tf.concat([qg, ql, v_qf, v_qt, wi, wo], axis=1), self.K_q)
        v_kcl = tf.reduce_mean(tf.abs(tf.concat([v_p, v_q], axis=1)))
        self.add_metric(self.mean(v_kcl)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_kcl)
        
        return inputs
    
    
class KVLLayer(Layer):
    '''
    Computes KVL violations from the provided cycle basis and branch variables
    Branch vars:  [c, s]
    Cycle basis:  cyinds, cysigns, cyrows
    '''   
    def __init__(self, cyinds, cysigns, cyrows, eta=0, name='kvl'):
        super(KVLLayer, self).__init__()
        self.cyinds = tf.convert_to_tensor(cyinds)
        self.cysigns = tf.convert_to_tensor(cysigns)
        self.cyrows = tf.convert_to_tensor(cyrows)
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        c, s = inputs
        c = tf.gather(c, self.cyinds, axis=1)
        s = self.cysigns*tf.gather(s, self.cyinds, axis=1)
        rc = tf.RaggedTensor.from_value_rowids(tf.transpose(c), self.cyrows)
        rs = tf.RaggedTensor.from_value_rowids(tf.transpose(s), self.cyrows)
        v_kvl = tf.transpose(tf.reduce_sum(tf.math.angle(tf.complex(rc, rs)), axis=1))
        v_kvl = tf.reduce_mean(tf.abs(v_kvl))
        self.add_metric(self.mean(v_kvl)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_kvl)
        
        return inputs


class PyLayer(Layer):
    '''
    Computes Pythagorean violations from the provided branch variables
    Branch vars:  [c, s, wi, wo]
    '''
    def __init__(self, py_from_ind, py_to_ind, eta=0, name='py'):
        super(PyLayer, self).__init__()
        self.py_from_ind = tf.convert_to_tensor(py_from_ind, dtype=tf.int32)
        self.py_to_ind = tf.convert_to_tensor(py_to_ind, dtype=tf.int32)
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        c, s, wi, wo = inputs
        v = tf.concat([wi, wo], axis=1)
        v_f = tf.expand_dims(tf.gather(v, self.py_from_ind, axis=1), axis=2)
        v_t = tf.expand_dims(tf.gather(v, self.py_to_ind, axis=1), axis=2)
        v_ft = tf.math.reduce_prod(tf.concat([v_f, v_t], axis=2), axis=2)
        v_py = tf.reduce_mean(tf.abs(c*c+s*s-v_ft))
        self.add_metric(self.mean(v_py)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_py)
        
        return inputs
    
    
class GLimitLayer(Layer):
    '''
    Computes generator limit violations using the provided set points
    Set points:  [pr, qg] or [pg, pr, qg]
    Limits:  [U, L]
    '''    
    def __init__(self, U, L, eta=0, name='glim'):
        super(GLimitLayer, self).__init__()
        self.U = U
        self.L = L
        self.act = ReLU()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        v = tf.concat(inputs, axis=1) # handles both [pr, qg] and [pg, pr, qg]
        v_l = self.act(self.L-v)
        v_u = self.act(v-self.U)
        v_glim = tf.reduce_mean(v_l+v_u)
        self.add_metric(self.mean(v_glim)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_glim)
        
        return inputs
    

class VLimitLayer(Layer):
    '''
    Computes voltage limit violations using the provided set points
    Voltages:  [wo] or [wi, wo]
    Limits:  [U, L]
    '''    
    def __init__(self, U, L, eta=0, name='vlim'):
        super(VLimitLayer, self).__init__()
        self.U = U
        self.L = L
        self.act = ReLU()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        v = tf.concat(inputs, axis=1) # handles both [wo] and [wi, wo]
        v_l = self.act(self.L-v)
        v_u = self.act(v-self.U)
        v_vlim = tf.reduce_mean(v_l+v_u)
        self.add_metric(self.mean(v_vlim)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_vlim)
        
        return inputs
    
    
class VALimitLayer(Layer):
    '''
    Computes voltage angle limit violations using the provided branch variables
    Branch vars:  [c, s]
    Limits:  [U, L]
    ''' 
    def __init__(self, U, L, eta=0, name='valim'):
        super(VALimitLayer, self).__init__()
        self.U = U
        self.L = L
        self.act = ReLU()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        c, s = inputs
        v = tf.math.angle(tf.complex(c, s))
        v_l = self.act(self.L-v)
        v_u = self.act(v-self.U)
        v_valim = tf.reduce_mean(v_l+v_u)
        self.add_metric(self.mean(v_valim)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_valim)
        
        return inputs
    
    
class TLimitLayer(Layer):
    '''
    Computes thermal limit violations using the provided flow variables
    Flows:  [v_pf, v_pt, v_qf, v_qt]
    Limits:  [U, L]
    '''
    def __init__(self, U_f, U_t, eta=0, name='tlim'):
        super(TLimitLayer, self).__init__()
        self.U_f = U_f
        self.U_t = U_t
        self.act = ReLU()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        v_pf, v_pt, v_qf, v_qt = inputs
        v_f = self.act(v_pf**2+v_qf**2-self.U_f)
        v_t = self.act(v_pt**2+v_qt**2-self.U_t)
        v_tlim = tf.reduce_mean(v_f+v_t)
        self.add_metric(self.mean(v_tlim)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_tlim)
        
        return inputs
    

class PFLimitLayer(Layer):
    '''
    Computes voltage angle limit violations using the provided set points
    Uses pfind variable to align qg with [pg, pr], in case synchrophasors were filtered out
    Set points:  [pg, pr, qg]
    Limits:  [U, L]
    '''
    def __init__(self, U, L, pfind, eta=0, name='pflim'):
        super(PFLimitLayer, self).__init__()
        self.U = U
        self.L = L
        self.pfind = tf.convert_to_tensor(pfind, dtype=tf.int32)
        self.act = ReLU()
        self.eta = tf.Variable(eta, dtype=tf.float32, trainable=False)
        self.mean = Mean(name=name)
        self.conname = name
        
    def call(self, inputs):
        pg, pr, qg = inputs
        v = tf.math.angle(tf.complex(tf.concat([pg, pr], axis=1), tf.gather(qg, self.pfind, axis=1)))
        v_l = self.act(self.L-v)
        v_u = self.act(v-self.U)
        v_pflim = tf.reduce_mean(v_l+v_u)
        self.add_metric(self.mean(v_pflim)) # batch accumulation is handled by Keras
        self.add_loss(self.eta*v_pflim)
        
        return inputs