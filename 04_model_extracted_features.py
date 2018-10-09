#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:42:52 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script uses the extractes features (number of misspelled words, word types
[percent nouns, verbs, adjectives], ) in multinomial  

"""

## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('read data')
score = pd.read_csv('score.csv')
num_misspelled = pd.read_csv('num_misspelled.csv')
word_type_pc = pd.read_csv('word_type_pc.csv')
sent_word_count = pd.read_csv('sent_word_count.csv')
sent_analysis = pd.read_csv('sent_analysis.csv')
tokens_str = pd.read_csv('tokens_str.csv')

nan_rows = tokens_str[tokens_str['tokens_str'].isnull()]
nan_rows =  nan_rows.index.values

score = score.drop(score.index[nan_rows])
num_misspelled = num_misspelled.drop(num_misspelled.index[nan_rows])
word_type_pc = word_type_pc.drop(word_type_pc.index[nan_rows])
sent_word_count = sent_word_count.drop(sent_word_count.index[nan_rows])
sent_analysis = sent_analysis.drop(sent_analysis.index[nan_rows])
tokens_str = tokens_str.drop(tokens_str.index[nan_rows])

features = pd.concat([num_misspelled, word_type_pc, sent_word_count, sent_analysis],axis=1)

features = features.drop(columns=['other', 'sent_statement','num_misspelled','noun','adjective','verb'])
score = np.ravel(score,order='F')

print('prepare feature model')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, score, test_size=0.2,random_state=40)

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

print('run feature model')
clf_features = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_features.fit(X_train, y_train)

y_predicted = clf_features.predict(X_test)

accuracy_features, precision_features, recall_features, f1_features = get_metrics(y_test, y_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_features, precision_features,recall_features, f1_features))


print('evaluate & plot tf-idf model')
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

cm = confusion_matrix(y_test, y_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3','4','5','6','7','8','9'], normalize=True, title='Confusion matrix')
plt.savefig('Confusion_Matrix_features.png', bbox_inches='tight')

### ROC, AUC features
#from sklearn.metrics import roc_curve, roc_auc_score
#
### Computing false and true positive rates
#rating_test_1 = np.where(y_test == 1)[0]
#rating_pre_1 = np.where(y_predicted == 1)[0]
#rating_1 = np.concatenate((rating_test_1, rating_pre_1), axis=0)
#
#y_test_1 = y_test[rating_1]
#y_predicted_1 = y_predicted[rating_1]
#y_predicted_1_idx = y_predicted_1 !=1 #[y_predicted_1 ~= 1]
#y_predicted_1[y_predicted_1_idx] = 0
#
#fpr, tpr,_ = roc_curve(y_test_1,y_predicted_1,drop_intermediate=False)
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
