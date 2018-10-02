#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:08:19 2018

@author: Johannes Rennig
@project: SituScore for Insight Data Science Toronto 18C
@description: consulting project for Altus Assessments, Toronto, ON, Canada

This script loads in the data set cleaned from statements reported in French. 
After tokenizing each statement using NLTK the script performs a spell check 
on each word using thr Python library spellchecker and saves the number of 
misspelled words per statement. The script also deletes every misspelled word
but this is not recommended since it takes very long. Saving the corrected 
statements into a data frame and then into a csv breaks the kernel. 

"""
## Clean up

## Import packages
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from spellchecker import SpellChecker

## Read in data
print('read data')
data = pd.read_csv('data_clean_1.csv')

## NLP pre-processing
print('- NLP pre-processing -')

# Find number of words, sentences, spelling errors
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

## Spellcheck
print('spell check and delete misspelled words')

spell = SpellChecker()

num_misspelled = []
#tokens_corr = []
tokens_corr = pd.DataFrame()
for idx in range(0, len(data)):
    
    print(f'Corr: Item {idx+1} of {len(data)}')
    tokens_c = data['tokens_pre_clean'].iloc[idx]
    misspelled = list(spell.unknown(data['tokens_pre_clean'].iloc[idx]))
    num_misspelled.append(len(misspelled))   
    tokens_c = [e for e in tokens_c if e not in misspelled]
    if len(tokens_c) < 0:
        tokens_c = 'nan'

    tokens_c = list(tokens_c)    
    tokens_corr.append(tokens_c)
    tokens_corr = tokens_corr.append(tokens_c,ignore_index=True)


num_misspelled = pd.DataFrame(num_misspelled)
num_misspelled.columns = ['num_misspelled']
num_misspelled.to_csv('num_misspelled.csv')

tokens_corr = pd.DataFrame(tokens_corr)
tokens_corr.columns = ['tokens_corr']
tokens_corr.to_csv('tokens_corr.csv')
