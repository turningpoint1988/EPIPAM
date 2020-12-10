import os, sys, argparse
from Bio import SeqIO
import pandas as pd
import numpy as np
import random
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


###################### Input #######################
def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-c", dest="cell", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')

    return parser.parse_args()


parse = get_args()
CELL = parse.cell
NAME = parse.name
RESAMPLE_TIME = 10
RESAMPLE_TIME_T = 1 # 1 or 10
PROMOTER_LEN = 2000 #promoter
ENHANCER_LEN = 3000 #enhancer
ROOT = os.path.dirname(os.path.abspath('__file__'))
CHROMESIZE = {}
with open(os.path.join(ROOT, 'hg19/chromsize')) as f:
    for i in f:
        line_split = i.strip().split()
        CHROMESIZE[line_split[0]] = int(line_split[1])


def split(test_chroms):
    pairs = pd.read_csv(CELL+'/%s.csv' % NAME)
    pairs = pairs.iloc[:, [1, 5, 3, 4, 7, 10, 8, 9, 6]]
    n_sample = pairs.shape[0]
    test_index = []
    train_index = []
    for i in range(n_sample):
        chr = pairs.iloc[i, 0]
        if chr in test_chroms:
            test_index.append(i)
        else:
            train_index.append(i)
    # for train data
    pairs_train = pairs.iloc[train_index]
    pairs_train_pos = pairs_train[pairs_train['label'] == 1]
    pos_num = pairs_train_pos.shape[0]
    pairs_train_neg = pairs_train[pairs_train['label'] == 0]
    neg_num = pairs_train_neg.shape[0]
    if pos_num * RESAMPLE_TIME > neg_num:
        sample_num = neg_num
    else:
        sample_num = pos_num * RESAMPLE_TIME
    rand_index = list(range(neg_num))
    np.random.shuffle(rand_index)
    sample_index = rand_index[:sample_num]
    pairs_train_neg = pairs_train_neg.iloc[sample_index]
    pairs_train_filter = pd.concat([pairs_train_pos, pairs_train_neg])
    # for test
    pairs_test = pairs.iloc[test_index]
    pairs_test_pos = pairs_test[pairs_test['label'] == 1]
    pos_num = pairs_test_pos.shape[0]
    pairs_test_neg = pairs_test[pairs_test['label'] == 0]
    neg_num = pairs_test_neg.shape[0]
    if pos_num * RESAMPLE_TIME_T > neg_num:
        sample_num = neg_num
    else:
        sample_num = pos_num * RESAMPLE_TIME_T
    rand_index = list(range(neg_num))
    np.random.shuffle(rand_index)
    sample_index = rand_index[:sample_num]
    pairs_test_neg = pairs_test_neg.iloc[sample_index]
    pairs_test_filter = pd.concat([pairs_test_pos, pairs_test_neg])
    #save
    pairs_train_filter.to_csv(CELL + '/pairs_train.csv', index=False)
    pairs_test_filter.to_csv(CELL + '/pairs_test.csv', index=False)


def resize_location(original_location, resize_len, chr):
    chr_end = CHROMESIZE[chr]
    start = int(original_location[0])
    end = int(original_location[1])
    original_len = end - start
    if original_len < resize_len:
        start_update = start - np.ceil((resize_len-original_len)/2)
    elif original_len > resize_len:
        start_update = start + np.ceil((original_len - resize_len)/2)
    else:
        start_update = start
    rand_int = np.random.randint(-100, 100)
    resize_start = int(start_update) - rand_int
    if resize_start < 1:
        resize_start = 1
    resize_end = resize_start + resize_len
    if resize_end > chr_end:
        resize_end = chr_end
        resize_start = resize_end - resize_len
    return str(resize_start), str(resize_end)


def resize_location_fix(original_location, resize_len, chr):
    chr_end = CHROMESIZE[chr]
    start = int(original_location[0])
    end = int(original_location[1])
    original_len = end - start
    if original_len < resize_len:
        start_update = start - np.ceil((resize_len-original_len)/2)
    elif original_len > resize_len:
        start_update = start + np.ceil((original_len - resize_len)/2)
    else:
        start_update = start
    resize_start = int(start_update)
    if resize_start < 1:
        resize_start = 1
    resize_end = resize_start + resize_len
    if resize_end > chr_end:
        resize_end = chr_end
        resize_start = resize_end - resize_len
    return str(resize_start), str(resize_end)


def augment(infile, outfile):
    fout = open(outfile, 'w')
    file = open(infile)
    title = file.readline()
    fout.write(title)
    for line in file:
        line = line.strip().split(',')
        if line[-1] == '0':
            original_location = (line[1], line[2])
            resized_location = resize_location_fix(original_location, ENHANCER_LEN, line[0])
            fout.write(','.join([line[0], resized_location[0], resized_location[1], line[3]]) + ',')
            original_location = (line[5], line[6])
            resized_location = resize_location_fix(original_location, PROMOTER_LEN, line[4])
            fout.write(','.join([line[4], resized_location[0], resized_location[1], line[7], line[-1]]) + '\n')
        else:
            for j in range(0, RESAMPLE_TIME):
                original_location = (line[1], line[2])
                resized_location = resize_location(original_location, ENHANCER_LEN, line[0])
                fout.write(','.join([line[0], resized_location[0], resized_location[1], line[3]]) + ',')
                original_location = (line[5], line[6])
                resized_location = resize_location(original_location, PROMOTER_LEN, line[4])
                fout.write(','.join([line[4], resized_location[0], resized_location[1], line[7], line[-1]]) + '\n')
    file.close()
    fout.close()


def augment_fix(infile, outfile):
    fout = open(outfile, 'w')
    file = open(infile)
    title = file.readline()
    fout.write(title)
    for line in file:
        line = line.strip().split(',')
        original_location = (line[1], line[2])
        resized_location = resize_location_fix(original_location, ENHANCER_LEN, line[0])
        fout.write(','.join([line[0], resized_location[0], resized_location[1], line[3]]) + ',')
        original_location = (line[5], line[6])
        resized_location = resize_location_fix(original_location, PROMOTER_LEN, line[4])
        fout.write(','.join([line[4], resized_location[0], resized_location[1], line[7], line[-1]]) + '\n')
    file.close()
    fout.close()


def sentence2word(str_set):
    word_seq=[]
    for sr in str_set:
        tmp=[]
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq


def word2num(wordseq,tokenizer,MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num(str_set,tokenizer,MAX_LEN):
    wordseq=sentence2word(str_set)
    numseq=word2num(wordseq,tokenizer,MAX_LEN)
    return numseq


def get_tokenizer():
    f= ['a','c','g','t']
    c = itertools.product(f,f,f,f,f,f)
    res=[]
    for i in c:
        temp=i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res=np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null']=0
    return tokenizer


def get_data(enhancers, promoters):
    tokenizer = get_tokenizer()
    MAX_LEN = 3000
    X_en = sentence2num(enhancers, tokenizer, MAX_LEN)
    MAX_LEN = 2000
    X_pr = sentence2num(promoters, tokenizer, MAX_LEN)

    return X_en, X_pr


def one_hot(sequence_dict, chrom, start, end):
    seq = str(sequence_dict[chrom].seq[start:end])

    return seq


def encoding(sequence_dict, filename):
    file = open(CELL + '/' + filename)
    file.readline()
    seqs_1 = []
    seqs_2 = []
    label = []
    for line in file:
        line = line.strip().split(',')
        seqs_1.append(one_hot(sequence_dict, line[0], int(line[1]), int(line[2])))
        seqs_2.append(one_hot(sequence_dict, line[4], int(line[5]), int(line[6])))
        label.append(int(line[-1]))
    x_en_tr, x_pr_tr = get_data(seqs_1, seqs_2)

    return x_en_tr, x_pr_tr, np.array(label)


def main():
    """Split for training and testing data"""
    np.random.seed(1234)
    index = ['chr' + str(i + 1) for i in range(23)]
    index[22] = 'chrX'
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(os.path.join(ROOT, 'hg19/hg19.fa')), 'fasta'))
    for cv in range(8):
        test_chroms = index[cv * 3:(cv + 1) * 3]
        split(test_chroms)
        """Augment training data"""
        infile = CELL + '/pairs_train.csv'
        outfile = CELL + '/pairs_train_augment.csv'
        augment(infile, outfile)
        x_en_tr, x_pr_tr, y_tr = encoding(sequence_dict, 'pairs_train_augment.csv')
        np.savez(CELL + '/data' + '/%s_train_fold%d.npz' % (NAME, cv),
                 X_en_tr=x_en_tr, X_pr_tr=x_pr_tr, y_tr=y_tr)
        infile = CELL + '/pairs_test.csv'
        outfile = CELL + '/pairs_test_augment.csv'
        augment_fix(infile, outfile)
        x_en_te, x_pr_te, y_te = encoding(sequence_dict, 'pairs_test_augment.csv')
        np.savez(CELL + '/data' + '/%s_test_fold%d.npz' % (NAME, cv),
                 X_en_te=x_en_te, X_pr_te=x_pr_te, y_te=y_te)

"""RUN"""
if __name__ == "__main__": main()
