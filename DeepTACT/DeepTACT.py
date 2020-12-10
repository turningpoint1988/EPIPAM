#!/usr/bin/env python
#keras version: keras-1.2.0

import sys
import os, re
import argparse
import random
import datetime
import numpy as np
import os.path as osp
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score

from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.regularizers import l1, l2
from keras import initializers
from keras.callbacks import Callback


######################## GPU Settings #########################
os.environ["CUDA_VISIBLE_DEVICES"]="0"

########################### Input #############################
def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-c", dest="cell", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default='P-E')

    return parser.parse_args()


parse = get_args()
CELL = parse.cell
NAME = parse.name
ENHANCER_LEN = 3000 # or 1000
PROMOTER_LEN = 2000 # or 1000
root = osp.dirname(osp.abspath('__file__'))


######################## Initialization #######################
NUM_SEQ = 4
NUM_ENSEMBL = 8


########################### Training ##########################
# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        #self.init = initializations.get('normal')
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #self.W = self.init((input_shape[-1],))
        self.W = K.variable(self.init((input_shape[-1], 1)))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        M = K.tanh(x)
        alpha = K.dot(M, self.W)
        alpha = K.squeeze(alpha, -1)

        ai = K.exp(alpha)
        weights = ai/(K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weights = K.expand_dims(weights, -1)
        weighted_input = x * weights
        out = K.sum(weighted_input, axis=1)
        return K.tanh(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def model_def():
    enhancers = Input(shape=(ENHANCER_LEN, 4))
    promoters = Input(shape=(PROMOTER_LEN, 4))

    enhancer_conv_layer = Conv1D(filters=1024, kernel_size=40, padding="valid", activation='relu')(enhancers)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters=1024, kernel_size=40, padding="valid", activation='relu')(promoters)
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer)

    merge_layer = Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
    merge_layer_bn = BatchNormalization()(merge_layer)
    merge_layer_d = Dropout(0.5)(merge_layer_bn)

    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(merge_layer_d)
    l_att = AttLayer()(l_lstm)
    l_att_bn = BatchNormalization()(l_att)
    l_att_bn_d = Dropout(0.5)(l_att_bn)

    dense1 = Dense(925)(l_att_bn_d)
    dense1_bn = BatchNormalization()(dense1)
    dense1_bn_r = Activation('relu')(dense1_bn)
    dense1_bn_r_d = Dropout(0.5)(dense1_bn_r)
    preds = Dense(1, activation='sigmoid')(dense1_bn_r_d)

    model = Model([enhancers, promoters], preds)
    return model


class roc_callback(Callback):
    def __init__(self, val_data=None, cv=0):
        super(roc_callback, self).__init__()
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.cv = cv
        self.best_auc = 0.
        self.best_aupr = 0.

    def getresult(self):
        return self.best_auc, self.best_aupr

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en, self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        if (auc_val+aupr_val) > (self.best_auc+self.best_aupr):
            self.best_auc = auc_val
            self.best_aupr = aupr_val

        print('\r auc_val: %s ' %str(round(auc_val, 4)))
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def bagging(t):
    ## load data: sequence
    train_data = np.load(CELL+'/data/{}_train_fold{}.npz'.format(NAME, '_all'))
    X_en_tr = train_data['enhancer']
    X_pr_tr = train_data['promoter']
    y_tr = train_data['label']
    test_data = np.load(CELL+'/data/{}_test_fold{}.npz'.format(NAME, '_inter'))
    X_en_te = test_data['enhancer']
    X_pr_te = test_data['promoter']
    y_te = test_data['label']
    auc_best = 0
    prauc_best = 0
    for trail in range(5):
        model = model_def()
        model.summary()
        print('compiling...')
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        back = roc_callback(val_data=[X_en_te, X_pr_te, y_te], cv=t)
        print('fitting...')
        model.fit([X_en_tr, X_pr_tr], y_tr, epochs=10, batch_size=100, callbacks=[back])
        auc, prauc = back.getresult()
        if (auc_best + prauc_best) < (auc + prauc):
            auc_best = auc
            prauc_best = prauc
            model.save_weights(osp.join(root, 'model/specificModel/%sModel%d.h5' % (NAME, t)))
    ######
    if t == 7:
        model.load_weights(osp.join(root, 'model/specificModel/%sModel%d.h5' % (NAME, t)))
        y_pred = model.predict([X_en_te, X_pr_te])
        y_pred = np.asarray([y[0] for y in y_pred])
        y_real = np.asarray([y for y in y_te])
        with open(osp.join(root, 'model/%s_record%d.txt' % (NAME, t)), 'w') as f:
            for i in range(len(y_pred)):
                f.write('{}\t{}\n'.format(y_real[i], y_pred[i]))
    return auc_best, prauc_best


def train():
    file_name = osp.join(root, 'model/%s_record.txt' % NAME)
    f = open(file_name, 'w')
    f.write('fold\tAUC\tAUPRC\n')
    for t in range(NUM_ENSEMBL):
        best_auc, best_prauc = bagging(t)
        f.write("{}\t{:.3f}\t{:.3f}\n".format(t, best_auc, best_prauc))
        f.flush()
    f.close()


if __name__ == '__main__': train()

