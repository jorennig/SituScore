## SituScore
## Insight Data Science Project 2018 Toronto

This is the repository of my Insight Data Science project SituScore (Session 18CTO). The project is a consulting project for Altus Assessments, Toronto, ON, Canada. 

The data comes from a situational judgement test (CASPer) designed to select the ideal candidates for medical professions. The aim of the test is to assess qualifications that are necessary to successfully praction as a medical professional beyond traditional measures of academic success, like school grades or IQ tests. Each applicant takes the test online and has to watch videos of 12 scenarios displaying conflicting social situations and answers several questions in written form within a 5 minutes time window. Each response is then evaluated by a human rater and assessed on a scale between 1 to 9 (1 = poor situational awareness, 9 = high situational awareness). 

The data set consisted of 500 000 answers and the corresponding ratings. First, all answers in French had to be identified and were deleted (~ 7% of the data). Next, a spellcheck was performed and all misspelled words were deleted and counted per answer. Then, all answers were parsed through a general NLP pre-processing pipeline using NLTK including transforming all letters to lower case, removal of stopp words and lemmatization. In a next step PoS (part of sentence tagging) was performed and the percentage of nouns, verbs and adjectives was calculated. 

In a first analysis step, a sentiment analysis was conducted using the VADER package from NLTK. For each statement, 

First, all features derived from the the previous steps (pre-processing, PoS tagging, sentiment analysis) 

In a next analysis step, a tf-idf (term frequency inverse document frequency) model was implemented in a multinomial logistic regression with 9 categories (situational awareness scores from 1 to 9). This model classified with an overall accuracy of 
