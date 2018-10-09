#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:08:19 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script loads in the data set cleaned from statements reported in French. 

First it creates and saves a histogramm of the distribution of scores of the
whole sample.

Then it performs a general NLP preprocessing with the statements including
the following steps: 
- tokenization
- all words to lower case

The next analysis step is a sentiment analysis done with the NLTK package VADER. 
In a first pass a whole statement is anlyzed, then each sentence gets analyzed
to create an average sentiment score and the dispersion of sentiment ratings 
across a statement caluclating the standard deviation. A further sentiment
analysis calculates the percentage of emotional words per statement.

The next part of the script removes all stopwords and condicts a PoS tagging to
find the percentage of nouns, adjectives and verbs used in each statement.

Finally the script saves the pre-processed statements as a list to dataframe
for later modeling with tf-idf and bag of words techniques.

"""
## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

## Read in data
print('read data')
data = pd.read_csv('data_clean_1.csv')
#data = data[np.isfinite(data['score'])] # Remove NaN in 'score'

#num_sub = pd.DataFrame(data['applicantUserId'].unique())
#num_rater = pd.DataFrame(data['raterUserId'].unique())
#num_scene = pd.DataFrame(data['scenarioId'].unique())
#num_test = pd.DataFrame(data['testId'].unique())
#num_resp = pd.DataFrame(data['responseId'].unique())

# Plot distribution of scores
scores = np.unique(data['score'])
min_score = min(data['score'])
max_score = max(data['score'])

plt.hist(data['score'], bins=len(scores))
plt.title('Histogram Scores')
plt.xlabel('Scores')
plt.ylabel('Count')
plt.savefig('Histo_Scores.png', bbox_inches='tight')
plt.close()

## NLP pre-processing
print('- NLP pre-processing -')

# Tokenize and find number of words
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
data['tokens_pre_clean'] = data['answers'].apply(tokenizer.tokenize)

data['num_tokens_pre'] = data['tokens_pre_clean'].apply(len)

## Standardize text - all lower case
print('standardize text - all lower case')

def text_lower(df, text_field):
    df[text_field] = df[text_field].str.lower()
    return df

data = text_lower(data, 'answers')

data['tokens_pre_clean'] = data['answers'].apply(tokenizer.tokenize)

## Sentiment analysis and sentence count
print('sentiment analysis')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import nltk.data
tokenizer_sent = nltk.data.load('tokenizers/punkt/english.pickle')

sent_count = []
sent_statement = []
sent_sentence_mean = []
sent_sentence_std = []

for idx in range(0, len(data)):
    
    print(f'SE: Item {idx+1} of {len(data)}')
    text_c = data['answers'].iloc[idx]

    sent_statement_c = sid.polarity_scores(text_c)
    sent_statement.append(abs(sent_statement_c['compound']))
    
    ## Sentences to tokens
    sent_tokens = tokenizer_sent.tokenize(text_c)
    
    sent_sentence = []
    idx_s = 0
    for sent_c in sent_tokens:
        sent_sentence_c = sid.polarity_scores(sent_c)
        #sent_sentence.append(sent_sentence_c)
        sent_sentence.append(abs(sent_sentence_c['compound']))
        idx_s = idx_s + 1
    
    sent_sentence_mean.append(np.mean(sent_sentence))
    sent_sentence_std.append(np.std(sent_sentence))

data['sent_statement'] = sent_statement
data['sent_sentence_mean'] = sent_sentence_mean
data['sent_sentence_std'] = sent_sentence_std

sent_analysis = data.loc[:, ['sent_statement','sent_sentence_mean','sent_sentence_std']]
sent_analysis.to_csv('sent_analysis.csv',index=False)

## Remove stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords_add = ['would','could','can','u','o','e','m','n','t']
stopwords = stopwords + stopwords_add

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import operator
import itertools

find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]

tokens_clean = []
word_type_pc = []
sent_word_count = []

for tok in range(0, len(data)):
    
    print(f'L-Key: Item {tok+1} of {len(data)}')
    tokens_c = data['tokens_pre_clean'].iloc[tok]
    tokens_c = [token for token in tokens_c if token not in stopwords] # Remove stop words
    tokens_c = [lemmatizer.lemmatize(token) for token in tokens_c] ## Lemmatize each statement
    tokens_clean.append(tokens_c)
    
    pos_tag = nltk.pos_tag(tokens_c)
        
    pos_tag_list = list(map(operator.itemgetter(1), pos_tag))
    
    nn = find(pos_tag_list,['NN','NNP','NNPS','NNS','FW'])
    nn = len(list(itertools.chain(*nn)))
    ad = find(pos_tag_list,['JJ','JJR','JJS','RB','RBR','RBS'])
    ad = len(list(itertools.chain(*ad)))
    vb = find(pos_tag_list,['VB','VBD','VBG','VBN','VBP','VBZ'])
    vb = len(list(itertools.chain(*vb)))
    
    word_type = np.array([nn,ad,vb])
    
    word_type = list(word_type/len(pos_tag_list))
    other = sum(word_type)/len(pos_tag_list)
    word_type.append(other)
    word_type_pc.append(list(word_type))

    sent_word_tot = []
    for word in range(0, len(tokens_c)):
        word_c = tokens_c[word]
        sent_word_c = sid.polarity_scores(word_c)
        sent_word_c = sent_word_c['compound']
        sent_word_tot.append(abs(sent_word_c))
    
    sent_word_tot = np.array(sent_word_tot)
    sent_word_count_c = len([i for i in sent_word_tot if i>0.02])
    
    if len(tokens_c) > 0:
        sent_word_count_c = sent_word_count_c/len(tokens_c)
    else:
        sent_word_count_c = 'nan'
        
    sent_word_count.append(sent_word_count_c)
    
    
word_type_pc = pd.DataFrame(word_type_pc)
word_type_pc.columns = ['noun','adjective','verb','other']
word_type_pc.to_csv('word_type_pc.csv',index=False)

sent_word_count = pd.DataFrame(sent_word_count)
sent_word_count.columns = ['sent_word_pc']
sent_word_count.to_csv('sent_word_count.csv',index=False)

# Remodel tokens back to strings
tokens_str = []
for idx in range(0, len(tokens_clean)):
    print(f'Str: Item {idx+1} of {len(tokens_clean)}')
    tc = tokens_clean[idx]
    tcr = ' '.join(tc)
    tokens_str.append(tcr)
    
list_corpus = tokens_str

tokens_str_df = pd.DataFrame(tokens_str)
tokens_str_df.columns = ['tokens_str']
tokens_str_df.to_csv('tokens_str.csv',index=False)

# To avoid having to load all data for the models the score column is saved separately
score = data.loc[:, ['score']]
score.to_csv('score.csv',index=False)
