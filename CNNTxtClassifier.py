#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:09:11 2017

@author: nz
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model


class CNNTxtClassifier:
    def __init__(self, train_data = None, train_target = None, test_data = None, test_target = None, MAX_NB_WORDS = 20000, MAX_DOC_LEN = 500, NUM_OUTPUT_UNITS = 1, EMBEDDING_DIM = 100, FILTER_SIZES = [2, 3, 4],NUM_FILTERS = 64, DROP_OUT = 0.5,PRETRAINED_WORD_VECTOR = None, LAM = 0.01):
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_DOC_LEN = MAX_DOC_LEN
        self.NUM_OUTPUT_UNITS = NUM_OUTPUT_UNITS #number of output units
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.FILTER_SIZES = FILTER_SIZES
        self.NUM_FILTERS = NUM_FILTERS
        self.DROP_OUT = DROP_OUT
        self.PRETRAINED_WORD_VECTOR = PRETRAINED_WORD_VECTOR
        self.LAM = LAM
    
    def tokenize(self):
        self.tokenizer = Tokenizer(num_words = self.MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(self.train_data)
        self.voc = self.tokenizer.word_index
        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_data)
        # pad all sequences into the same length 
        # if a sentence is longer than maxlen, pad it in the right
        # if a sentence is shorter than maxlen, truncate it in the right
        self.train_padded_sequences = pad_sequences(self.train_sequences, maxlen = self.MAX_DOC_LEN, padding = 'post', truncating = 'post')
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_data)
        self.test_padded_sequences = pad_sequences(self.test_sequences, maxlen = self.MAX_DOC_LEN, padding = 'post', truncating = 'post')
    
    def cnn_model(self):
        main_input = Input(shape=(self.MAX_DOC_LEN,), \
                       dtype='int32', name='main_input')
    
        if self.PRETRAINED_WORD_VECTOR is not None:
            embed_1 = Embedding(input_dim = self.MAX_NB_WORDS + 1, \
                        output_dim = self.EMBEDDING_DIM, \
                        input_length = self.MAX_DOC_LEN, \
                        weights=[self.PRETRAINED_WORD_VECTOR],\
                        trainable=False,\
                        name='embedding')(main_input)
        else:
            embed_1 = Embedding(input_dim = self.MAX_NB_WORDS + 1, \
                        output_dim = self.EMBEDDING_DIM, \
                        input_length = self.MAX_DOC_LEN, \
                        name='embedding')(main_input)
        # add convolution-pooling-flat block
        conv_blocks = []
        total_num_filters = 0
        for f in self.FILTER_SIZES:
            conv = Conv1D(filters = self.NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
            conv = MaxPooling1D(self.MAX_DOC_LEN - f + 1, name='max_'+str(f))(conv)
            conv = Flatten(name='flat_'+str(f))(conv)
            conv_blocks.append(conv)
            total_num_filters += self.NUM_FILTERS

        z = Concatenate(name='concate')(conv_blocks)
        drop = Dropout(rate = self.DROP_OUT, name='dropout')(z)

        dense = Dense(total_num_filters, activation='relu',\
                    kernel_regularizer = l2(self.LAM),name='dense')(drop)
        preds = Dense(self.NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
        model = Model(inputs=main_input, outputs=preds)
    
        model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"]) 
    
        return model
    

    def early_stop(self, monitor = 'val_loss', patience = 0, verbose = 2, mode = 'min'):
        earlyStopping = EarlyStopping(monitor = monitor, patience = patience, verbose = verbose, mode = mode)
        return earlyStopping
    def check_point(self, BEST_MODEL_FILEPATH, monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max'):
        return ModelCheckpoint(BEST_MODEL_FILEPATH, monitor = monitor, \
                             verbose = verbose, save_best_only = save_best_only, mode = mode)
