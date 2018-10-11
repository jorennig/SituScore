#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:41:52 2018

@author: beauchamplab
"""

## Import packages
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import nltk

print('read data')
score = pd.read_csv('score.csv')
tokens_str = pd.read_csv('tokens_str.csv')

#num_misspelled = pd.read_csv('num_misspelled.csv')
#word_type_pc = pd.read_csv('word_type_pc.csv')
#sent_word_count = pd.read_csv('sent_word_count.csv')
#sent_analysis = pd.read_csv('sent_analysis.csv')

nan_rows = tokens_str[tokens_str['tokens_str'].isnull()]
nan_rows =  nan_rows.index.values

score = score.drop(score.index[nan_rows])
tokens_str = tokens_str.drop(tokens_str.index[nan_rows])

#num_misspelled = num_misspelled.drop(num_misspelled.index[nan_rows])
#word_type_pc = word_type_pc.drop(word_type_pc.index[nan_rows])
#sent_word_count = sent_word_count.drop(sent_word_count.index[nan_rows])
#sent_analysis = sent_analysis.drop(sent_analysis.index[nan_rows])

#features = pd.concat([num_misspelled, word_type_pc, sent_word_count, sent_analysis],axis=1)

## Summarize even scores
score[score['score'] == 2] = 1
score[score['score'] == 3] = 1

score[score['score'] == 4] = 2
score[score['score'] == 5] = 2
score[score['score'] == 6] = 2

score[score['score'] == 7] = 3
score[score['score'] == 8] = 3
score[score['score'] == 9] = 3

list_labels = score['score'].tolist()
list_corpus = tokens_str['tokens_str'].tolist()

# Vectorize
print('prepare model')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.winter):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=10)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    return plt


######
print('prepare Bag of Words model')

def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                random_state=40)
X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)

print('run Bag of Words model')
clf_counts = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_counts.fit(X_train_counts, y_train)

y_predicted_counts = clf_counts.predict(X_test_counts)

accuracy_counts, precision_counts, recall_counts, f1_counts = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_counts, precision_counts,recall_counts, f1_counts))

print('evaluate & plot Bag of Words model')
cm = confusion_matrix(y_test, y_predicted_counts)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3'], normalize=True, title='Confusion matrix')
plt.savefig('Confusion_Matrix_Counts_Reduced.png', bbox_inches='tight')


### Computing false and true positive rates
#fpr, tpr,_ = roc_curve(y_test,y_predicted_counts,drop_intermediate=False)
#
#import matplotlib.pyplot as plt
#plt.figure()
### Adding the ROC
#plt.plot(fpr, tpr, color='red',
# lw=2, label='ROC curve')
### Random FPR and TPR
#plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
### Title and label
#plt.xlabel('FPR')
#plt.ylabel('TPR')
#plt.title('ROC curve')
#plt.show()









