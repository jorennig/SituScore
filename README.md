## SituScore
## Insight Data Science Project 2018 Toronto

This is the repository of my Insight Data Science project SituScore (Session 18CTO). The project is a consulting project for Altus Assessments, Toronto, ON, Canada. 

The data comes from a situational judgement test (CASPer) designed to select the ideal candidates for medical professions. The aim of the test is to assess qualifications that are necessary to successfully praction as a medical professional beyond traditional measures of academic success, like school grades or IQ tests. Each applicant takes the test online and has to watch videos of 12 scenarios displaying conflicting social situations and answers several questions in written form within a 5 minutes time window. Each response is then evaluated by a human rater and assessed on a scale between 1 to 9 (1 = poor situational awareness, 9 = high situational awareness). 

The data set consisted of 500 000 answers and the corresponding ratings. First, all answers in French had to be identified and were deleted (~ 7% of the data). Next, a spellcheck was performed and all misspelled words were deleted and counted per answer. Then, all answers were parsed through a general NLP pre-processing pipeline using NLTK including transforming all letters to lower case, removal of stopp words and lemmatization. In a next step PoS (part of sentence tagging) was performed and the percentage of nouns, verbs and adjectives was calculated. During this pre-processing procedure 4 text-derived features were created: number of misspelled words, percentage of nouns, verbs and adjectives.

In a first analysis step, a sentiment analysis was conducted using the VADER package from NLTK. For each statement, a sentiment score was derived. VADER gives a compound sentiment score ranging from -1 (primarily negative sentiment content) to 1 (primarily positive sentiment content). For later analysis, the absolute value of this analysis was stored since the direction of the compound value was not of interest. The sentiment analysis was first applied to the entire statement and then repeated per sentence to calculate an average sentiment score and the dispersion (standard deviation) of emotional content across the answer. Additionally, the sentiment analysis was conducted wordwise to determine the percentage of emotional words per answer. The whole analysis provided 4 different sentiment based features: sentiment score of full answer, average sentiment score (across sentences), dispersion of sentiment score (standard deviation across sentences) and the pecentage of emotional words per statement.

First, all features derived from the the previous steps (pre-processing, PoS tagging, sentiment analysis) were used in a multinomial logistic regression with 9 categories (situational awareness scores from 1 to 9). This model did not perform all too well and classified the different ratings witn an overall accuracy of 7% (see overview of all classification metrics in the file model_performance_situscore.csv in this repository).

Confusion matrix of the feature-based model with all 9 categories:

<img src="https://github.com/jorennig/SituScore/blob/master/Confusion_Matrix_features_full_scale.png" alt="CM" width="300" height="300">

In a next analysis step, a tf-idf (term frequency inverse document frequency) model was implemented in a multinomial logistic regression with 9 categories. This model classified with an overall accuracy of 
