#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:49:55 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script loads in the origonal data and idenfifies statement written in 
French and removes them from the data set. This is done by identifying french
stopwords from the NLTK stopword corpus of several European languages. 

The data is saved to csv under the name 'data_clean_1.csv'.

"""

## Import packages
import pandas as pd
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize

# Load data
print('Load data')
data = pd.read_csv('rating-metrics.scores.csv')

## Idenitfy and remove French statements
print('Identify french')

lang_ans = []
for idx in range(0, len(data)):
    text = data['answers'].iloc[idx]
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    
    languages_ratios = {}    
    for language in ['english','french']:
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements) # language "score"
        
    most_rated_language = max(languages_ratios, key=languages_ratios.get)
    lang_ans.append(most_rated_language)

data['lang'] = lang_ans
lang_ans = pd.DataFrame({'lang':lang_ans})

percent_french = (len(lang_ans[lang_ans['lang'] == 'french'])/len(lang_ans))*100
percent_english = (len(lang_ans[lang_ans['lang'] == 'english'])/len(lang_ans))*100

data = data[data['lang'] == 'english']
data = data.drop(['lang'], axis=1)

del language, lang_ans, languages_ratios, most_rated_language, text, tokens, words

data.to_csv('data_clean_1.csv')
