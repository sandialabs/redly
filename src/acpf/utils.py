#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
from .models import *


#############
## evaluation
#############

def get_single_run(model, dataset):
    '''
    Function to get test and prediction dataframes for a single run
    '''
    incols = np.concatenate(dataset.model_data['incols'])
    outcols = np.concatenate(dataset.model_data['outcols'])

    x = np.concatenate([batch[0].numpy() for batch in iter(dataset.test)])
    y = np.concatenate([batch[1].numpy() for batch in iter(dataset.test)])[:,:-1] # remove b_flag
    y_ = model.predict(x)[:,np.arange(len(outcols))]
    
    xdf = pd.DataFrame(x, columns=incols)
    ydf = pd.DataFrame(y, columns=outcols)
    ydf_ = pd.DataFrame(y_, columns=outcols)
    
    return xdf, ydf, ydf_


def get_runs(folder, name_prefix, runs, ilayers, dataset, 
             constraints={}, alpha=0, dropout=0):
    '''
    Function to get losses, test, and prediction dataframes for
    each run of a series
    '''
    histories = []
    dfts = []
    mdatas = []
    for run in range(runs):
        # clear graph state
        tf.compat.v1.reset_default_graph()

        # load model and training data for run
        model, model_data, mask_data, histdict = load_run(
            folder, '%s_run_%d'%(name_prefix, run), ilayers, 
            constraints=constraints, alpha=alpha, dropout=dropout)

        histories.append(histdict)
        dfts.append(get_single_run(model, dataset))
        mdatas.append(mask_data)
        
    return histories, dfts, mdatas


def get_counts(mdata):
    '''
    Function to count the number of active weights/neurons from mask dictionary
    '''
    # get layer info
    lname = list(mdata.keys())[0]
    lname = lname[:lname.rfind('_')]
    final_layers = [k for k in mdata.keys() if not k.split('_')[-1].isdigit()]
    num_internal_layers = len(mdata) - len(final_layers)
    
    # build mask list
    mask_list = []
    for i in range(num_internal_layers):
        mask_list.append(mdata['%s_%d'%(lname,i)]['mask'])
    # penultimate layers are concatenated together
    mask_list.append(np.concatenate([mdata[k]['mask'] 
                                     for k in final_layers], axis=1))
    
    # weight counts
    total_weight_count = [m.shape[0]*m.shape[1] for m in mask_list]
    active_weight_count = [np.sum(m>0) for m in mask_list]
    
    # total neuron and input counts
    input_count = mask_list[0].shape[0]
    total_neuron_count = [m.shape[0] for m in mask_list[1:]]
    
    # active neuron count
    active_neuron_count = []
    for l in range(num_internal_layers):
        # lower check - matmul to see which neurons receive signal from input
        lower = np.ones(input_count)
        for i in range(l+1):
            lower = np.matmul(lower, mask_list[i])
            
        # upper check - trace each neuron to see if it contributes to output
        upper = np.zeros(total_neuron_count[l])
        for n in range(total_neuron_count[l]):
            impulse = np.zeros(total_neuron_count[l])
            impulse[n] = 1
            for i in range(l+1, len(mask_list)):
                impulse = np.matmul(impulse, mask_list[i])
            if np.sum(impulse) > 0:
                upper[n] = 1
        
        # both checks need to pass for neuron to count as active
        active_neuron_count.append(np.sum((upper*lower)>0))
        
    return np.sum(total_weight_count), np.sum(active_weight_count), \
        np.sum(total_neuron_count), np.sum(active_neuron_count)


def get_single_summary(history, dft, model_data):
    '''
    Function to get summary dataframe for a single run
    '''
    total_weights = []
    active_weights = []
    total_neurons = []
    active_neurons = []
    pr_errors = []
    qg_errors = []
    wo_errors = []
    c_errors = []
    s_errors = []
    errors = []
    
    edf = dft[1]-dft[2]
    cols = edf.columns
        
    # errors
    pr_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][0]]].values)))
    qg_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][1]]].values)))
    wo_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][2]]].values)))
    c_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][3]]].values)))
    s_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][4]]].values)))
    errors.append(np.mean(np.abs(edf.values)))
        
    sdf = pd.DataFrame({'pr_errors':pr_errors, 'qg_errors':qg_errors,
                        'wo_errors':wo_errors, 'c_errors':c_errors,
                        's_errors':s_errors, 'total_errors':errors})
    return sdf


def get_summary(histories, dfts, mdatas, model_data):
    '''
    Function to get summary dataframe for series of runs
    '''
    total_weights = []
    active_weights = []
    total_neurons = []
    active_neurons = []
    pr_errors = []
    qg_errors = []
    wo_errors = []
    c_errors = []
    s_errors = []
    errors = []
    
    for (history, dft, mdata) in zip(histories, dfts, mdatas):
        edf = dft[1]-dft[2]
        cols = edf.columns

        # weight/neuron counts
        total_weight_counts, active_weight_counts, \
            total_neuron_counts, active_neuron_counts = get_counts(mdata)
        total_weights.append(total_weight_counts)
        active_weights.append(active_weight_counts)
        total_neurons.append(total_neuron_counts)
        active_neurons.append(active_neuron_counts)
        
        # errors
        pr_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][0]]].values)))
        qg_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][1]]].values)))
        wo_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][2]]].values)))
        c_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][3]]].values)))
        s_errors.append(np.mean(np.abs(edf[cols[model_data['oinds'][4]]].values)))
        errors.append(np.mean(np.abs(edf.values)))
        
    sdf = pd.DataFrame({'total_weights':total_weights, 'active_weights':active_weights,
                        'total_neurons':total_neurons, 'active_neurons':active_neurons,
                        'pr_errors':pr_errors, 'qg_errors':qg_errors,
                        'wo_errors':wo_errors, 'c_errors':c_errors,
                        's_errors':s_errors, 'total_errors':errors})
    return sdf
        
        
#############
## plotting
#############

def plot_run_loss(histories, keys=[], ld_int=0, ld_prog=0, 
                  figsize=(8,8), fontsize=12, titlesize=14):
    '''
    Function to plot selected loss metrics
    '''
    # use first run to set up keys
    keyfilter = ['loss', 'val_loss']
    for key in keys:
        keyfilter.append(key)
        keyfilter.append('val_'+key)
    keysf = [k for k in histories[0].keys() if k in keyfilter and len(histories[0][k])>0]
    numepochs = len(histories[0]['loss'])
    
    # num rows/columns for plot grid
    nrows = int(np.sqrt(len(histories)))
    ncols = int(np.ceil(len(histories)/nrows))
    
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)
    
    for k, histdict in enumerate(histories):
        if nrows > 1:
            i = k//ncols
            j = k % ncols
            ax = axs[i, j]
        elif ncols > 1:
            ax = axs[k]
        else:
            ax = axs
        
        epochs = []
        metrics = []
        ks = []
        ktypes = []
        for key in keysf:
            metrics.append(np.log(histdict[key]))
            if key.startswith('val_'):
                kname = key[4:]
                ktype = 'val'
            else:
                kname = key
                ktype = 'train'
            ks.append(np.repeat(kname, numepochs))
            ktypes.append(np.repeat(ktype, numepochs))
            epochs.append(np.arange(numepochs))
        df = pd.DataFrame({'epoch':np.concatenate(epochs), 
                           'metric':np.concatenate(metrics),
                           'loss':np.concatenate(ks),
                           'type':np.concatenate(ktypes)})
        
        sns.lineplot(ax=ax, x='epoch', y='metric', hue='loss', style='type', data=df)
        if ld_int > 0:
            x = ld_int
            dx = ld_int
            while x < numepochs:
                ax.axvline(x=x, linestyle='--', color='b', lw=1, alpha=.5)
                dx += ld_prog
                x += dx
        
        ax.set_title('Run %s'%k, fontsize=fontsize)
        if k > 0:
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    
    fig.supxlabel('Epoch', fontsize=fontsize)
    fig.supylabel('Log Loss', fontsize=fontsize)
    plt.suptitle('Losses', fontsize=titlesize)
    plt.tight_layout()
    plt.show()


def plot_run_errors(sdf, figsize=(8,8), fontsize=12, titlesize=14):
    '''
    Function to plot MAE of each variable type over all runs
    '''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    plt.plot(np.arange(len(sdf)), sdf['pr_errors'], label='pr', 
             alpha=.2, linewidth=3, marker='.', markersize=10)
    plt.plot(np.arange(len(sdf)), sdf['qg_errors'], label='qg', 
             alpha=.2, linewidth=3, marker='.', markersize=10)    
    plt.plot(np.arange(len(sdf)), sdf['wo_errors'], label='wo', 
             alpha=.2, linewidth=3, marker='.', markersize=10)
    plt.plot(np.arange(len(sdf)), sdf['c_errors'], label='c', 
             alpha=.2, linewidth=3, marker='.', markersize=10)    
    plt.plot(np.arange(len(sdf)), sdf['s_errors'], label='s', 
             alpha=.2, linewidth=3, marker='.', markersize=10)
    plt.plot(np.arange(len(sdf)), sdf['total_errors'], label='total', 
             linewidth=3, marker='.', color='b', markersize=10)

    ax.tick_params(labelsize=fontsize)
    plt.xlabel('run', fontsize=fontsize)
    plt.ylabel('error', fontsize=fontsize)
    plt.title('Mean Absolute Errors', fontsize=titlesize)
    ax.legend(bbox_to_anchor=(1,1), fontsize=fontsize)
        
    plt.show()    


#############
## save/load
#############

def save_run(folder, name_prefix, model, model_data, histdict, mask_data={}):
    '''
    Function to save model weights and training history
    Generates four sets of files:
    1. tensorflow model save data (.data-xxx-xxx, .index, and checkpoint files)
    2. model_data dictionary
    3. mask_data dictionary
    4. training history
    '''
    # confirm directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # save model weights
    model.save_weights(os.path.join(folder, name_prefix))
    
    # save model_data dictionary
    np.save(os.path.join(folder, '%s_model_data.npy'%name_prefix), model_data)
    
    # save mask_data dictionary
    np.save(os.path.join(folder, '%s_mask_data.npy'%name_prefix), mask_data)
    
    # save training history
    np.save(os.path.join(folder, '%s_history.npy'%name_prefix), histdict)


def load_run(folder, name_prefix, ilayers, constraints={}, alpha=0, dropout=0):
    '''
    Function to load model weights and training history
    '''
    # load model_data dictionary
    model_data = np.load(os.path.join(folder, '%s_model_data.npy'%name_prefix), 
                         allow_pickle=True).item()
    
    # load mask_data dictionary
    mask_data = np.load(os.path.join(folder, '%s_mask_data.npy'%name_prefix), 
                        allow_pickle=True).item()
    
    # load training history
    histdict = np.load(os.path.join(folder, '%s_history.npy'%name_prefix),
                       allow_pickle=True).item()
    
    # build model and load weights
    model = buildACPFModel(ilayers, model_data, constraints=constraints, 
                           mask_data=mask_data, alpha=alpha, dropout=dropout)
    model.load_weights(os.path.join(folder, name_prefix)).expect_partial()
    
    return model, model_data, mask_data, histdict


def export_run_to_json(folder, name_prefix, ilayers, constraints={}, alpha=0, dropout=0):
    '''
    Function to load model and convert to json for verification
    '''   
    model, _, _, _ = load_run(folder, name_prefix, ilayers, constraints=constraints, alpha=alpha, dropout=dropout)
    layers_json = []
    
    # internal layers
    for i in range(0, len(ilayers)):
        layer = model.get_layer('mdense_%d'%i)
        d = {}
        d['W'] = layer.weights[0].numpy().tolist()
        d['b'] = layer.weights[1].numpy().tolist()
        d['mask'] = layer.weights[2].numpy().tolist()
        d['activation'] = 'relu'
        d['alpha'] = alpha
        layers_json.append(d)
        
    # output layers
    otypes = ['pr', 'qg', 'wo', 'c', 's']
    W = []
    b = []
    mask = []
    for o in otypes:
        layer = model.get_layer('mdense_%s'%o)
        W.append(layer.weights[0].numpy())
        b.append(layer.weights[1].numpy())
        mask.append(layer.weights[2].numpy())
    W_json = np.hstack(W)
    b_json = np.hstack(b)
    mask_json = np.hstack(mask)
    d = {}
    d['W'] = W_json.tolist()
    d['b'] = b_json.tolist()
    d['activation'] = 'identity'  # TODO implement use of conditional relu in updated model
    d['mask'] = mask_json.tolist()
    layers_json.append(d)
    
    filename = os.path.join(folder, '%s_model_data.json'%name_prefix)
    with open(filename, 'w') as f:
        json.dump(layers_json, f)
    
    
