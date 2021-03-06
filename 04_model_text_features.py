"""
Created on Thu Sep 27 13:42:52 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script uses the extractes text features (number of misspelled words, 
word types [percent nouns, verbs, adjectives], ) and all metrics from the 
sentiment analyses in two versions of a multinomial logistic regression:
- 9 categories (full rating scale 1-9)
- 3 categories (rating scale summarized: [1,2,3] = 1, [4,5,6] = 2, [7,8,9] = 3)

"""

## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import itertools
from sklearn.metrics import confusion_matrix

## Load and prepare data
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
features = features.drop(columns=['other'])

labels = np.ravel(score,order='F')

## Define functions
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


def plot_confusion_matrix(cm, classes,normalize=False,title='',cmap=plt.cm.winter):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=10)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=20)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    return plt


# Model with full rating scale
print('prepare and run feature model with full rating scale')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state=40)

clf_features = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_features.fit(X_train, y_train)

y_predicted = clf_features.predict(X_test)

accuracy_features, precision_features, recall_features, f1_features = get_metrics(y_test, y_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_features, precision_features,recall_features, f1_features))

print('evaluate & plot model')
cm = confusion_matrix(y_test, y_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3','4','5','6','7','8','9'], normalize=True, title='')
plt.savefig('Confusion_Matrix_features_full_scale.png', bbox_inches='tight')


## Summarize scores
score[score['score'] == 2] = 1
score[score['score'] == 3] = 1

score[score['score'] == 4] = 2
score[score['score'] == 5] = 2
score[score['score'] == 6] = 2

score[score['score'] == 7] = 3
score[score['score'] == 8] = 3
score[score['score'] == 9] = 3

label = np.ravel(score,order='F')

# Model with summarized rating scale
print('prepare and run feature model with summarized rating scale')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state=40)

clf_features = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_features.fit(X_train, y_train)

print('evaluate & plot')
y_predicted = clf_features.predict(X_test)

accuracy_features, precision_features, recall_features, f1_features = get_metrics(y_test, y_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_features, precision_features,recall_features, f1_features))

cm = confusion_matrix(y_test, y_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3'], normalize=True, title=' ')
plt.savefig('Confusion_Matrix_features_summarized.png', bbox_inches='tight')
