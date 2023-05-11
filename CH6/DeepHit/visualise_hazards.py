_EPSILON = 1e-04

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
# import sys

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepHit import Model_DeepHit
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score, c_t_index


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here
    return data



##### MAIN SETTING
OUT_ITERATION               = 1

data_mode                   = 'SYNTHETIC_CROSSING' #METABRIC, SYNTHETIC
seed                        = 1234

EVAL_TIMES                  = [10] # evalution times (for C-index and Brier-Score)


##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (num_features)
    mask1, mask2            = used for cause-specific network (FCNet structure)
'''
if data_mode == 'SYNTHETIC':
    (x_dim), (data, time, label), (mask1, mask2, mask3) = impt.import_dataset_SYNTHETIC(norm_mode = 'standard')
    EVAL_TIMES  = [36]
elif data_mode == 'METABRIC':
    (x_dim), (data, time, label), (mask1, mask2, mask3) = impt.import_dataset_METABRIC(norm_mode = 'standard')
    EVAL_TIMES = [19]
elif data_mode == 'SYNTHETIC_CROSSING':
    (x_dim), (data, time, label), (mask1, mask2, mask3) = impt.import_dataset_SYNTHETIC_CROSSING(
        norm_mode='standard')
    EVAL_TIMES = [10]
else:
    print('ERROR:  DATA_MODE NOT FOUND !!!')

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]


in_path = data_mode + '/results_TD/'

if not os.path.exists(in_path):
    os.makedirs(in_path)


FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL2 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL3 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])  #Added results recording
FINAL4 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL5 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL6 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])

for out_itr in range(OUT_ITERATION):
    in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters.txt'
    in_parser = load_logging(in_hypfile)

    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']

    h_dim_shared                = in_parser['h_dim_shared']
    h_dim_CS                    = in_parser['h_dim_CS']
    num_layers_shared           = in_parser['num_layers_shared']
    num_layers_CS               = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn                = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn                = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn                = tf.nn.tanh
    else:
        print('Error!')


    initial_W                   = tf.contrib.layers.xavier_initializer()

    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    delta                       = in_parser['delta']  #for haz ranking loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma)) + 'd' + str('%02.0f' %(10*delta))


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                    'h_dim_CS'          : h_dim_CS,
                                    'num_layers_shared'    : num_layers_shared,
                                    'num_layers_CS'    : num_layers_CS,
                                    'active_fn'      : active_fn,
                                    'initial_W'         : initial_W }


    # for out_itr in range(OUT_ITERATION):
    print ('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ' d:' + str(delta) + ')' )
    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data,te_data, tr_time,te_time, tr_label,te_label,
     tr_mask1,te_mask1, tr_mask2,te_mask2, tr_mask3,te_mask3)  = \
        train_test_split(data, time, label, mask1, mask2, mask3, test_size=0.20, random_state=seed)

    (tr_data,va_data, tr_time,va_time, tr_label,va_label,
     tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3)  =\
        train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, test_size=0.20, random_state=seed)

    ##### PREDICTION & EVALUATION
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### PREDICTION
    pred = model.predict(te_data)

    import matplotlib.pyplot as plt

    ind1 = te_data[:, 0] < 0
    ind2 = te_data[:, 0] > 0
    pred1 = pred[ind1, 0, :]
    pred2 = pred[ind2, 0, :]

    xmin = [i for i in range(10)]
    xmax = [i for i in range(1, 11)]
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    h1 = pred1[0, :-2] / (1 - np.append(0, np.cumsum(pred1[0, :-3])))
    h2 = pred2[0, :-2] / (1 - np.append(0, np.cumsum(pred2[0, :-3])))
    plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', label=r'$\hat{\alpha}(t\vert Z_{i,1}=0)$', alpha=1)
    plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', label=r'$\hat{\alpha}(t\vert Z_{i,1}=1)$', alpha=1)

    for i in range(pred1.shape[0]):
        h1 = pred1[i, :-2] / (1 - np.append(0, np.cumsum(pred1[i, :-3])))
        plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', alpha=0.01)

    for i in range(pred2.shape[0]):
        h2 = pred2[i, :-2] / (1 - np.append(0, np.cumsum(pred2[i, :-3])))
        plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', alpha=0.01)

    x = [i + 0.5 for i in range(10)]
    ticks = [str(i) for i in range(1, 11)]
    plt.xticks(x, ticks, minor=False)
    plt.yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8), (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8))

    plt.xlabel('t')
    #plt.ylabel(r'$\alpha(t\vert z)$')
    plt.legend(loc='upper left')
    plt.savefig('images/beta_'+str(in_parser['beta']) + '_delta_' + str(in_parser['delta']) + '_' + str(out_itr) + '2.pdf')
    plt.close()

    #Now visualise the cumulative hazards
    H1 = np.cumsum(pred1[0, :-2] / (1 - np.append(0, np.cumsum(pred1[0, :-3]))))
    H2 = np.cumsum(pred2[0, :-2] / (1 - np.append(0, np.cumsum(pred2[0, :-3]))))
    plt.hlines(H2, xmin, xmax, colors='r', linestyles='solid', label=r'$\hat{H}(t\vert Z_{i,1}=0)$', alpha=1)
    plt.hlines(H1, xmin, xmax, colors='b', linestyles='solid', label=r'$\hat{H}(t\vert Z_{i,1}=1)$', alpha=1)

    for i in range(pred1.shape[0]):
        H1 = np.cumsum(pred1[i, :-2] / (1 - np.append(0, np.cumsum(pred1[i, :-3]))))
        plt.hlines(H1, xmin, xmax, colors='b', linestyles='solid', alpha=0.01)

    for i in range(pred2.shape[0]):
        H2 = np.cumsum(pred2[i, :-2] / (1 - np.append(0, np.cumsum(pred2[i, :-3]))))
        plt.hlines(H2, xmin, xmax, colors='r', linestyles='solid', alpha=0.01)

    x = [i + 0.5 for i in range(10)]
    ticks = [str(i) for i in range(1, 11)]
    plt.xticks(x, ticks, minor=False)
    plt.xlabel('t')
    plt.yticks((0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5), (0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5))
    plt.show()

    plt.legend(loc='upper left')
    plt.savefig('images/cum_beta_'+str(in_parser['beta']) + '_delta_' + str(in_parser['delta']) +'_' +str(out_itr)+ '2.pdf')
    plt.close()