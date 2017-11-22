#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:06:39 2017

@author: nz
"""
import pandas as pd
from CNNTxtClassifier import CNNTxtClassifier as cnn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


"""
load in data
"""
filename = 'amazon_review_large.csv'
amazon_large = pd.read_csv(filename, sep = ',',header = -1)
amazon_large = amazon_large.rename(columns = {0: 'label', 1: 'review'})
#update label to [0, 1]
reviews = amazon_large['review']
labels = amazon_large['label'] - 1

"""
split data into train and test datasets, the test dataset is used for comparing performace of all models.
"""
x_train, x_test, y_train, y_test = train_test_split(reviews,labels, test_size = 0.15)

"""
convert train and test datasets into ifidf forms
"""
def tfidf_transform(x_train, x_test):
    tfidf_vect = TfidfVectorizer(stop_words = 'english') #ngram_range = (1,3)
    dtm_train = tfidf_vect.fit_transform(x_train) #shape: (17000, 40260)
    dtm_test = tfidf_vect.transform(x_test) #shape: (3000, 40260) shape as a whole: (20000, 43652)
    return dtm_train, dtm_test

dtm_train, dtm_test = tfidf_transform(x_train, x_test)


"""
Multinomial Naive Bayes
"""

mnb_clf = MultinomialNB(alpha = 0.5)
mnb_clf.fit(dtm_train, y_train)
predictions = mnb_clf.predict(dtm_test)

def evaluate_accuracy(preditions, ground_truths):
    correct_count = 0
    for p, g in zip(predictions, ground_truths):
        if p == g:
            correct_count += 1
    return correct_count*1.0/len(ground_truths)

MNB_accuracy = evaluate_accuracy(predictions, y_test)
print("accuracy of MNB on test data:", MNB_accuracy) # 0.8093333333333333

"""
Support Vector Machine
"""

svm_clf = svm.LinearSVC()
svm_clf.fit(dtm_train, y_train)
predictions = svm_clf.predict(dtm_test)
SVM_accuracy = evaluate_accuracy(predictions, y_test)
print("accuracy of SVM on test data:", SVM_accuracy) #0.8376666666666667

"""
CNN
"""

MAX_NB_WORDS=40000
MAX_DOC_LEN=1000
EMBEDDING_DIM = 300
# filters on bigrams, trigrams, and 4-grams
FILTER_SIZES=[1, 2, 3, 4]
NUM_FILTERS = 64
DROP_OUT = 0.5
PRETRAINED_WORD_VECTOR = None
LAM = 0.01

BEST_MODEL_FILEPATH = 'best_model'


BATCH_SIZES = 64
NUM_EPOCHES = 20

cnn_clf = cnn(train_data = x_train, train_target = y_train, test_data = x_test,test_target = y_test, MAX_NB_WORDS = MAX_NB_WORDS, MAX_DOC_LEN = MAX_DOC_LEN, EMBEDDING_DIM = EMBEDDING_DIM, FILTER_SIZES = FILTER_SIZES, NUM_FILTERS = NUM_FILTERS, DROP_OUT = DROP_OUT, PRETRAINED_WORD_VECTOR = PRETRAINED_WORD_VECTOR, LAM = LAM)

cnn_clf.tokenize()
model = cnn_clf.cnn_model()
early_stop = cnn_clf.early_stop()
check_point = cnn_clf.check_point(BEST_MODEL_FILEPATH)
training = model.fit(cnn_clf.train_padded_sequences, cnn_clf.train_target, \
          batch_size = BATCH_SIZES, epochs= NUM_EPOCHES, \
          callbacks=[early_stop, check_point],\
          validation_split=0.2, verbose=2)
predictions = model.predict(cnn_clf.test_padded_sequences)

def evaluate_accuracy_cnn(predictions, ground_truths):
    correct_count = 0
    for p, g in zip(predictions, ground_truths):
        if p >= 0.5:
            t = 1
            if t == g:
                correct_count += 1
        else:
            t = 0
            if t == g:
                correct_count += 1
    return correct_count*1.0/len(predictions)

CNN_accuracy = evaluate_accuracy_cnn(predictions, cnn_clf.test_target) 
print("accuracy of CNN on test data:", CNN_accuracy) #0.8583333333333333

"""
output

Train on 13600 samples, validate on 3400 samples
Epoch 1/20
Epoch 00000: val_acc improved from -inf to 0.83676, saving model to best_model
744s - loss: 0.9688 - acc: 0.7190 - val_loss: 0.4068 - val_acc: 0.8368
Epoch 2/20
Epoch 00001: val_acc improved from 0.83676 to 0.85794, saving model to best_model
947s - loss: 0.3354 - acc: 0.8751 - val_loss: 0.3571 - val_acc: 0.8579
Epoch 3/20
Epoch 00002: val_acc improved from 0.85794 to 0.86559, saving model to best_model
965s - loss: 0.2007 - acc: 0.9362 - val_loss: 0.3865 - val_acc: 0.8656
Epoch 00002: early stopping
('accuracy of CNN on test data:', 0.8583333333333333)

"""

    
    
