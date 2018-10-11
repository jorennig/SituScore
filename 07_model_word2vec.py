"""
Created on Thu Sep 27 13:42:52 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script uses all relevant statements and vectorizes them using word2vec
in two versions of a multinomial logistic regression:
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

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

import gensim

word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

## Load and prepare data
print('read data')
score = pd.read_csv('score.csv')
tokens_str = pd.read_csv('tokens_str.csv')

nan_rows = tokens_str[tokens_str['tokens_str'].isnull()]
nan_rows =  nan_rows.index.values

score = score.drop(score.index[nan_rows])
tokens_str = tokens_str.drop(tokens_str.index[nan_rows])

list_labels = score['score'].tolist()

tokenizer = RegexpTokenizer(r'\w+')
tokens_clean = pd.DataFrame()
tokens_clean['tokens_clean'] = tokens_str['tokens_str'].apply(tokenizer.tokenize)

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


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = tokens_clean['tokens_clean'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)


# Model with full rating scale
print('prepare and run word2vec model with full rating scale')

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

X_train_count, count_vectorizer = cv(X_train)
X_test_count = count_vectorizer.transform(X_test)

print('run Bag of Words model')
clf_count = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_count.fit(X_train_count, y_train)

y_predicted_count = clf_count.predict(X_test_count)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_count)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision,recall, f1))

print('evaluate & plot tf-idf model')
cm = confusion_matrix(y_test, y_predicted_count)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3','4','5','6','7','8','9'], normalize=True, title='')
plt.savefig('Confusion_Matrix_word2vec_full_scale.png', bbox_inches='tight')


## Summarize to three categories
score[score['score'] == 2] = 1
score[score['score'] == 3] = 1

score[score['score'] == 4] = 2
score[score['score'] == 5] = 2
score[score['score'] == 6] = 2

score[score['score'] == 7] = 3
score[score['score'] == 8] = 3
score[score['score'] == 9] = 3

list_labels = score['score'].tolist()

# Model with summarized rating scale
print('prepare and run Bag of Words model with summarized rating scale')

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

X_train_count, count_vectorizer = cv(X_train)
X_test_count = count_vectorizer.transform(X_test)

print('run Bag of Words model')
clf_count = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_count.fit(X_train_count, y_train)

y_predicted_count = clf_count.predict(X_test_count)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_count)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision,recall, f1))

cm = confusion_matrix(y_test, y_predicted_count)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['1','2','3'], normalize=True, title='')
plt.savefig('Confusion_Matrix_BoW_summarized.png', bbox_inches='tight')
