#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import json

from src.acpf.losses import BMSELoss, BMSEMetric
from src.acpf.callbacks import TimeCallback, ESCallback, LDCallback
from src.acpf.datasets import ACPFDataset
from src.acpf.models import buildACPFModel
from src.acpf.utils import save_run, load_run, plot_run_loss, plot_run_errors, get_single_run, get_single_summary, export_run_to_json


def main(config, dofit=True):
    '''
    Train a single ACPF prediction model (no pruning)
    config:  configuration dictionary from acpf.configs
    dofit:   True to train a new model, False to load a previous model
    '''
    
    b_p = config['b_p']                   # boundary pt. fraction
    ilayers = config['ilayers']           # internal model layer widths
    alpha = config['alpha']               # LeakyReLU alpha parameter
    dropout = config['dropout']           # dropout rate
    rho = config['rho']                   # LD rho update parameter
    ld_int = config['ld_int']             # initial LD interval length
    ld_prog = config['ld_prog']           # LD arithmetic progression parameter
    batch = config['batch']               # batch size
    epochs = config['epochs']             # num. epochs
    patience = config['patience']         # patience for early stopping
    lr = config['lr']                     # learning rate  
    constraints = config['constraints']   # selected constraints
    
    # set up folder + name prefix for saving output
    folder = config['folder']
    name_prefix = '%s_b_%.2g_a_%.2g_d_%.2g_ldi_%d_ldp_%d_single'%(
        config['name'], b_p, alpha, dropout, ld_int, ld_prog)

    dataset = ACPFDataset(config, b_p=b_p, constraints=constraints)

    if dofit:
        dataset.verify()
        dataset.get_splits(batch=batch)
        dataset.save_splits(folder, name_prefix)
    else:
        dataset.load_splits(folder, name_prefix, batch=batch)
        dataset.verify(split='train')

    if dofit:
        # define model
        model = buildACPFModel(ilayers, dataset.model_data, constraints=dataset.constraints, 
                               alpha=alpha, dropout=dropout)
    
        # define callbacks/metrics
        opt = Adam(amsgrad=True, lr=lr)
        tcb = TimeCallback()
        escb = ESCallback(patience=patience)
        ldcb = LDCallback(opt, rho=rho, ld_int=ld_int, ld_prog=ld_prog)
        loss = BMSELoss(dataset.model_data)
        bmse = BMSEMetric(dataset.model_data)
    
        # compile + train
        model.compile(loss=loss, optimizer=opt, metrics=[bmse], 
                      steps_per_execution=len(dataset.train))
        
        history = model.fit(dataset.train, validation_data=dataset.val, 
                            epochs=epochs, callbacks=[tcb, escb, ldcb])
    
        histdict = history.history
        save_run(folder, name_prefix, model, dataset.model_data, histdict)
        export_run_to_json(folder, name_prefix, ilayers, constraints=constraints, 
                           alpha=alpha, dropout=dropout)        

    # clear graph state
    tf.compat.v1.reset_default_graph()
    
    # load run
    model, model_data, mask_data, histdict = load_run(
        folder, name_prefix, ilayers, constraints=constraints, alpha=alpha, dropout=dropout)

    print('Argmin_epoch: %d, argmin_loss: %.2e'%(np.argmin(histdict['val_loss']), 
                                                 np.min(histdict['val_loss'])))

    dft = get_single_run(model, dataset)
    sdf = get_single_summary(histdict, dft, dataset.model_data)
    print(sdf)

    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--case',
                        default='ieee_14',
                        choices=['ieee_14', 'ieee_118'],
                        type=str,
                        help='configuration')
    parser.add_argument('-f',
                        '--fit',
                        default=True,
                        type=bool,
                        help='True to train a new model, False to load an existing one')
    args = parser.parse_args()

    case = args.case
    if case=='ieee_14':
        with open('src/acpf/ieee_case14.json', 'r') as f:
            config = json.load(f)
    elif case=='ieee_118':
        with open('src/acpf/ieee_case118.json', 'r') as f:
            config = json.load(f)
    dofit = args.fit

    # physics run
    main(config, dofit=dofit)

    # non-physics run
    config2 = config.copy()
    config2['ld_int'] = config2['epochs']
    main(config2, dofit=dofit)