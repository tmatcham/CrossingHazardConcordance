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

        ### EVALUATION
        result1, result2 = np.zeros([num_Event, len(EVAL_TIMES)]), np.zeros([num_Event, len(EVAL_TIMES)])
        result3, result4 = np.zeros([num_Event, len(EVAL_TIMES)]), np.zeros([num_Event, len(EVAL_TIMES)])
        result5, result6 = np.zeros([num_Event, len(EVAL_TIMES)]), np.zeros([num_Event, len(EVAL_TIMES)])

        for t, t_time in enumerate(EVAL_TIMES):
            eval_horizon = int(t_time)

            if eval_horizon >= num_Category:
                print( 'ERROR: evaluation horizon is out of range')
                result1[:, t] = result2[:, t] = -1
            else:
                # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
                risk2 = np.cumsum(pred[:, :, :], axis=2)
                risk3 = pred / (1.00000000000000001 - risk2)
                for k in range(num_Event):
                    result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                    result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                    result3[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                    result4[k, t] = weighted_brier_score(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                    result5[k, t] = c_t_index(risk2[:,k,:], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon)  #C_td
                    result6[k, t] = c_t_index(risk3[:,k,:], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #C_alpha

        FINAL1[:, :, out_itr] = result1
        FINAL2[:, :, out_itr] = result2
        FINAL3[:, :, out_itr] = result3
        FINAL4[:, :, out_itr] = result4
        FINAL5[:, :, out_itr] = result5
        FINAL6[:, :, out_itr] = result6



        ### SAVE RESULTS
        row_header = []
        for t in range(num_Event):
            row_header.append('Event_' + str(t+1))

        col_header1 = []
        col_header2 = []
        col_header3 = []
        col_header4 = []
        col_header5 = []
        col_header6 = []


        for t in EVAL_TIMES:
            col_header1.append(str(t) + 'yr c_index')
            col_header2.append(str(t) + 'yr B_score')
            col_header3.append(str(t) + 'yr c_index_weighted')
            col_header4.append(str(t) + 'yr B_score_weighted')
            col_header5.append(str(t) + 'yr c_index_td')
            col_header6.append(str(t) + 'yr c_index_alpha')

        # c-index result
        df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)
        df1.to_csv(in_path + '/result_C_itr' + str(out_itr) + '.csv')

        # brier-score result
        df2 = pd.DataFrame(result2, index = row_header, columns=col_header2)
        df2.to_csv(in_path + '/result_BRIER_itr' + str(out_itr) + '.csv')

        # weighted c index
        df3 = pd.DataFrame(result3, index = row_header, columns=col_header3)
        df3.to_csv(in_path + '/result_WEIGHTED_C_itr' + str(out_itr) + '.csv')

        # weighted brier-score result
        df4 = pd.DataFrame(result4, index = row_header, columns=col_header4)
        df4.to_csv(in_path + '/result_WEIGHTED_BRIER_itr' + str(out_itr) + '.csv')

        # c td
        df5 = pd.DataFrame(result5, index = row_header, columns=col_header5)
        df5.to_csv(in_path + '/result_C_TD_itr' + str(out_itr) + '.csv')

        # c alpha
        df6 = pd.DataFrame(result6, index = row_header, columns=col_header6)
        df6.to_csv(in_path + '/result_C_ALPHA_itr' + str(out_itr) + '.csv')


        ### PRINT RESULTS
        print('========================================================')
        print('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ' d:' + str(delta) + ')' )
        print('SharedNet Parameters: ' + 'h_dim_shared = '+str(h_dim_shared) + ' num_layers_shared = '+str(num_layers_shared) + 'Non-Linearity: ' + str(active_fn))
        print('CSNet Parameters: ' + 'h_dim_CS = '+str(h_dim_CS) + ' num_layers_CS = '+str(num_layers_CS) + 'Non-Linearity: ' + str(active_fn))

        print('--------------------------------------------------------')
        print('- C-INDEX: ')
        print(df1)
        print('--------------------------------------------------------')
        print('-BRIER-SCORE: ')
        print(df2)
        print('--------------------------------------------------------')
        print('- WEIGHTED C-INDEX: ')
        print(df3)
        print('--------------------------------------------------------')
        print('- WEIGHTED BRIER-SCORE: ')
        print(df4)
        print('--------------------------------------------------------')
        print('- C-INDEX TD: ')
        print(df5)
        print('--------------------------------------------------------')
        print('- C-INDEX ALPHA: ')
        print(df6)

        print('========================================================')

import matplotlib.pyplot as plt

ind1 = te_data[:,0] < 0
ind2 = te_data[:,0] > 0
pred1 = pred[ind1,0,:]
pred2 = pred[ind2,0,:]

xmin = [i for i in range(10)]
xmax = [i for i in range(1,11)]

h1 = pred1[0,:-2] / (1 - np.append(0, np.cumsum(pred1[0,:-3])))
h2 = pred2[0,:-2] / (1 - np.append(0, np.cumsum(pred2[0,:-3])))
plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', label='Z=0', alpha=1)
plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', label='Z=1', alpha=1)

for i in range(pred1.shape[0]):
    h1 = pred1[i, :-2] / (1 - np.append(0, np.cumsum(pred1[i, :-3])))
    plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', alpha = 0.01)

for i in range(pred2.shape[0]):
    h2 = pred2[i, :-2] / (1 - np.append(0, np.cumsum(pred2[i, :-3])))
    plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', alpha = 0.01)

x = [i+0.5 for i in range(10)]
ticks = [str(i) for i in range(1,11)]
plt.xticks(x, ticks, minor = False)
plt.xlabel('t')
plt.ylabel(r'$h(t\vert z)$')
plt.legend(loc ='center left')





### FINAL MEAN/STD
# c-index result
df1_mean = pd.DataFrame(np.mean(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_std  = pd.DataFrame(np.std(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_mean.to_csv(in_path + '/result_CINDEX_FINAL_MEAN.csv')
df1_std.to_csv(in_path + '/result_CINDEX_FINAL_STD.csv')

# brier-score result
df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_std  = pd.DataFrame(np.std(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_mean.to_csv(in_path + '/result_BRIER_FINAL_MEAN.csv')
df2_std.to_csv(in_path + '/result_BRIER_FINAL_STD.csv')

# weighted c-index result
df1_mean = pd.DataFrame(np.mean(FINAL3, axis=2), index = row_header, columns=col_header1)
df1_std  = pd.DataFrame(np.std(FINAL3, axis=2), index = row_header, columns=col_header1)
df1_mean.to_csv(in_path + '/result_WEIGHTED_CINDEX_FINAL_MEAN.csv')
df1_std.to_csv(in_path + '/result_WEIGHTED_CINDEX_FINAL_STD.csv')

# weighted brier-score result
df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_std  = pd.DataFrame(np.std(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_mean.to_csv(in_path + '/result_WEIGHTED_BRIER_FINAL_MEAN.csv')
df2_std.to_csv(in_path + '/result_WEIGHTED_BRIER_FINAL_STD.csv')


### PRINT RESULTS
print('========================================================')
print('- FINAL C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL BRIER-SCORE: ')
print(df2_mean)
print('--------------------------------------------------------')
print('- FINAL WEIGHTED C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL WEIGHTED BRIER-SCORE: ')
print(df2_mean)
print('========================================================')