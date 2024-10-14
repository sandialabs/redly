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
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from .constraints import *
from .layers import *          


def buildACPFModel(ilayers, model_data, constraints={}, alpha=0, dropout=0, 
                   mask_data={}, lname='mdense', lamb=1e-10):
    '''
    Create PINN to compute ACPF state variables y from control variables x
    Assumes an additional b_flag variable to determine if y contributes to loss
    y* = NN(x); L(y,y*) = b_flag*L_{mse}(y,y*) + \sum_{phys} L_{phys}(x,y*)
    L_{phys} constraints are handled by Lagrangian Dual formalism
    External Lottery Ticket pruning loop is supported
    Note:  building using Keras API vs. subclassing appears to run faster
    '''
    # input dim
    in_dim = np.sum([len(cols) for cols in model_data['incols']])
    inputs = Input(shape=(in_dim,), dtype=tf.float32)
    
    # output dims
    pr_dim = len(model_data['outcols'][0])
    qg_dim = len(model_data['outcols'][1])
    wo_dim = len(model_data['outcols'][2])
    c_dim = len(model_data['outcols'][3])    
    s_dim = len(model_data['outcols'][4])
    
    # construct layers + dropouts
    layers = []
    dlayers = []
    for i, ilayer in enumerate(ilayers):
        if i==0:
            input_shape = (in_dim,)
        else:
            input_shape = (ilayers[i-1],)
        name = '%s_%d'%(lname,i)
        (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
            if name in mask_data else (None, None)
        layers.append(MDense(ilayer, name=name, mask=mask, init_weights=init_weights, 
                             input_shape=input_shape, use_bias=True, kernel_regularizer=l2(lamb)))
        dlayers.append(Dropout(dropout, input_shape=(ilayer,)))
    
    # output layers
    name = '%s_pr'%lname
    (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
        if name in mask_data else (None, None)
    pr_layer = MDense(pr_dim, name=name, mask=mask, init_weights=init_weights, 
                      input_shape=(ilayers[-1],), use_bias=True, kernel_regularizer=l2(lamb))
    name = '%s_qg'%lname
    (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
        if name in mask_data else (None, None)
    qg_layer = MDense(qg_dim, name=name, mask=mask, init_weights=init_weights, 
                      input_shape=(ilayers[-1],), use_bias=True, kernel_regularizer=l2(lamb))
    name = '%s_wo'%lname
    (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
        if name in mask_data else (None, None)
    wo_layer = MDense(wo_dim, name=name, mask=mask, init_weights=init_weights, 
                      input_shape=(ilayers[-1],), use_bias=True, kernel_regularizer=l2(lamb))
    name = '%s_c'%lname
    (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
        if name in mask_data else (None, None)
    c_layer = MDense(c_dim, name=name, mask=mask, init_weights=init_weights, 
                     input_shape=(ilayers[-1],), use_bias=True, kernel_regularizer=l2(lamb))
    name = '%s_s'%lname
    (mask, init_weights) = (mask_data[name]['mask'], mask_data[name]['weights']) \
        if name in mask_data else (None, None)
    s_layer = MDense(s_dim, name=name, mask=mask, init_weights=init_weights, 
                     input_shape=(ilayers[-1],), use_bias=True, kernel_regularizer=l2(lamb))
        
    # call
    pg = tf.gather(inputs, model_data['iinds'][0], axis=1)
    pl = tf.gather(inputs, model_data['iinds'][1], axis=1)
    ql = tf.gather(inputs, model_data['iinds'][2], axis=1)
    wi = tf.gather(inputs, model_data['iinds'][3], axis=1)
    
    h = tf.concat([pg, pl, ql, wi], axis=1)
    for layer, dlayer in zip(layers, dlayers):
        h = dlayer(relu(layer(h), alpha=alpha))
    pr = relu(pr_layer(h), alpha=alpha) # always >=0
    qg = qg_layer(h)
    wo = relu(wo_layer(h), alpha=alpha) # always >=0
    c = relu(c_layer(h), alpha=alpha) # always >=0
    s = s_layer(h)
    
    # compute flows
    v = tf.concat([wi, wo, c, s], axis=1)
    v_pf = tf.matmul(v, model_data['K_pf'])
    v_pt = tf.matmul(v, model_data['K_pt'])       
    v_qf = tf.matmul(v, model_data['K_qf'])
    v_qt = tf.matmul(v, model_data['K_qt'])
        
    # constraints; input layer elements are masked out for Keras compliance
    if 'kcl' in constraints:
        kcl_layer = KCLLayer(model_data['K_p'], model_data['K_q'], eta=constraints['kcl']['eta'],
                             name='kcl')
        [_, _, _, _, pr, qg, wo, v_pf, v_pt, v_qf, v_qt] = \
            kcl_layer([pg, pl, ql, wi, pr, qg, wo, v_pf, v_pt, v_qf, v_qt])
    if 'kvl' in constraints:
        kvl_layer = KVLLayer(model_data['cyinds'], model_data['cysigns'], 
                             model_data['cyrows'], eta=constraints['kvl']['eta'], name='kvl')
        [c, s] = kvl_layer([c, s])
    if 'py' in constraints:
        py_layer = PyLayer(model_data['py_from_ind'], model_data['py_to_ind'], 
                           eta=constraints['py']['eta'], name='py')
        [c, s, _, wo] = py_layer([c, s, wi, wo])
    if 'glim' in constraints:
        glim_layer = GLimitLayer(constraints['glim']['U'], constraints['glim']['L'], 
                                 eta=constraints['glim']['eta'], name='glim')
        [pr, qg] = glim_layer([pr, qg])
    if 'vlim' in constraints:
        vlim_layer = VLimitLayer(constraints['vlim']['U'], constraints['vlim']['L'], 
                                 eta=constraints['vlim']['eta'], name='vlim')
        [wo] = vlim_layer([wo])
    if 'valim' in constraints:
        valim_layer = VALimitLayer(constraints['valim']['U'], constraints['valim']['L'], 
                                   eta=constraints['valim']['eta'], name='valim')
        [c, s] = valim_layer([c, s])
    if 'tlim' in constraints:
        tlim_layer = TLimitLayer(constraints['tlim']['U_f'], constraints['tlim']['U_t'], 
                                 eta=constraints['tlim']['eta'], name='tlim')
        [v_pf, v_pt, v_qf, v_qt] = tlim_layer([v_pf, v_pt, v_qf, v_qt])
    if 'pflim' in constraints:
        pflim_layer = PFLimitLayer(constraints['pflim']['U'], constraints['pflim']['L'], 
                                   model_data['pfind'], eta=constraints['pflim']['eta'],
                                   name='pflim')
        [_, pr, qg] = pflim_layer([pg, pr, qg])

    outputs = tf.concat([pr, qg, wo, c, s, v_pf, v_pt, v_qf, v_qt], axis=1)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model