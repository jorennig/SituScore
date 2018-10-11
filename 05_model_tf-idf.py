"""
Created on Thu Sep 27 13:42:52 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script uses the extractes features (number of misspelled words, word types
[percent nouns, verbs, adjectives], ) in two versions of a multinomial logistic
regression:
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

## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

## Summarize to three categories
#score[score['score'] == 2] = 1
#score[score['score'] == 3] = 1
#
#score[score['score'] == 4] = 2
#score[score['score'] == 5] = 2
#score[score['score'] == 6] = 2
#
#score[score['score'] == 7] = 3
#score[score['score'] == 8] = 3
#score[score['score'] == 9] = 3

## Two categories
score[score['score'] == 2] = 1
score[score['score'] == 3] = 1
score[score['score'] == 4] = 1
score[score['score'] == 5] = 1
score[score['score'] == 6] = 0
score[score['score'] == 7] = 0
score[score['score'] == 8] = 0
score[score['score'] == 9] = 0

#num_misspelled = num_misspelled.drop(num_misspelled.index[nan_rows])
#word_type_pc = word_type_pc.drop(word_type_pc.index[nan_rows])
#sent_word_count = sent_word_count.drop(sent_word_count.index[nan_rows])
#sent_analysis = sent_analysis.drop(sent_analysis.index[nan_rows])

#features = pd.concat([num_misspelled, word_type_pc, sent_word_count, sent_analysis],axis=1)

list_labels = score['score'].tolist()
list_corpus = tokens_str['tokens_str'].tolist()

# Vectorize
print('prepare model')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(input):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(input)

    return train, tfidf_vectorizer

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

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

print('run tf-idf model')
clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf,recall_tfidf, f1_tfidf))

print('evaluate & plot tf-idf model')
import itertools
from sklearn.metrics import confusion_matrix

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

cm = confusion_matrix(y_test, y_predicted_tfidf)
fig = plt.figure(figsize=(10, 10))
#plot = plot_confusion_matrix(cm, classes=['1','2','3'], normalize=True, title='')
#plt.savefig('Confusion_Matrix_tfidf_reduced_ConSum.png', bbox_inches='tight')
#plot = plot_confusion_matrix(cm, classes=['1','2','3','4','5','6','7','8','9'], normalize=True, title='')
#plt.savefig('Confusion_Matrix_tfidf_full.png', bbox_inches='tight')
plot = plot_confusion_matrix(cm, classes=['1','2'], normalize=True, title='')
plt.savefig('Confusion_Matrix_tfidf_2cat.png', bbox_inches='tight')


### ROC, AUC if-idf
from sklearn.metrics import roc_curve, roc_auc_score

## Computing false and true positive rates
fpr, tpr,_ = roc_curve(y_test,y_predicted_tfidf,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
## Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
## Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
## Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
