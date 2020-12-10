#coding=UTF-8

# In[ ]:
import os
import os.path as osp
import h5py
import argparse
import numpy as np
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

os.environ["CUDA_VISIBLE_DEVICES"]="0"

MAX_LEN_en = 3000  # 1000
MAX_LEN_pr = 2000  # 1000
NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_model():
    enhancers = Input(shape=(MAX_LEN_en,))
    promoters = Input(shape=(MAX_LEN_pr,))

    emb_en = Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(enhancers)
    emb_pr = Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(promoters)

    enhancer_conv_layer = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    enhancer_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)

    merge_layer = Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    l_gru = Bidirectional(GRU(50, return_sequences=True))(dt)
    l_att = AttLayer(50)(l_gru)

    preds = Dense(1, activation='sigmoid')(l_att)

    model = Model([enhancers, promoters], preds)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


class roc_callback(Callback):
    def __init__(self, val_data=None, name=None, cv=0):
        super(roc_callback, self).__init__()
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name
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
        y_pred = np.array([y[0] for y in y_pred])
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


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-c", dest="cell", type=str, default='P-E')
    parser.add_argument("-n", dest="name", type=str, default='P-E')

    return parser.parse_args()


parse = get_args()
cell = parse.cell
name = parse.name
root = osp.dirname(osp.abspath('__file__'))

index = ['chr'+str(i+1) for i in range(23)]
index[22] = 'chrX'

file_name = osp.join(root, 'model/%s_record.txt' % name)
f = open(file_name, 'w')
f.write('fold\tAUC\tAUPRC\n')
Data_dir = cell + '/data'
for cv in range(8):
    test_chroms = ' '.join(index[cv * 3:(cv + 1) * 3])
    print("#######################################")
    print("The current test data is {}".format(test_chroms))
    print("#######################################")
    train = np.load(Data_dir+'/{}_train_fold{}.npz'.format(name, cv))
    test = np.load(Data_dir + '/{}_test_fold{}.npz'.format(name, cv))
    X_en_tr, X_pr_tr, y_tr = train['X_en_tr'], train['X_pr_tr'], train['y_tr']
    X_en_te, X_pr_te, y_te = test['X_en_te'], test['X_pr_te'], test['y_te']
    auc_best = 0
    prauc_best = 0
    for trail in range(5):
        model = get_model()
        model.summary()
        print('Traing %s cell line specific model ...'%name)

        back = roc_callback(val_data=[X_en_te, X_pr_te, y_te], name=name, cv=cv)
        history = model.fit([X_en_tr, X_pr_tr], y_tr, epochs=10, batch_size=100, callbacks=[back])
        auc, prauc = back.getresult()
        if (auc_best + prauc_best) < (auc + prauc):
            auc_best = auc
            prauc_best = prauc
            model.save_weights(osp.join(root, 'model/specificModel/%sModel%d.h5' % (name, cv)))
    f.write("{}\t{:.3f}\t{:.3f}\n".format(cv, auc_best, prauc_best))
    f.flush()
    #####
    if cv == 7:
        model.load_weights(osp.join(root, 'model/specificModel/%sModel%d.h5' % (name, cv)))
        y_pred = model.predict([X_en_te, X_pr_te])
        y_pred = np.asarray([y[0] for y in y_pred])
        y_real = np.asarray([y for y in y_te])
        with open(osp.join(root, 'model/%s_record%d.txt' % (name, cv)), 'w') as f1:
            for i in range(len(y_pred)):
                f1.write('{}\t{}\n'.format(y_real[i], y_pred[i]))
f.close()
