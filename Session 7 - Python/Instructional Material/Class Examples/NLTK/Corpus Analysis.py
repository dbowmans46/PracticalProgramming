#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:44:54 2024

@author: doug
"""



###############################################################################
#                                                                             #
#                           Frequency Distribution                            #
#                                                                             #
###############################################################################

import nltk

dracula_filepath = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Books/Dracula.txt"
with open(dracula_filepath) as fileHandle:
    dracula_string = fileHandle.read()
dracula_tokens = nltk.word_tokenize(dracula_string)
draculaText = nltk.Text(dracula_tokens)
draculaFreqDist = draculaText.vocab()




###############################################################################
#                                                                             #
#                           Bag of Words with Count                           #
#                                                                             #
###############################################################################

import sklearn.feature_extraction.text.CountVectorizer

# TODO: Load in the IMDB data into a dataframe
# TODO: Get all the words as the initial vocab
# TODO: Remove stop words from vocab
# TODO: Remove punctuation and small words from vocab
# TODO: Utilize the CountVectorizer with the vocab to get counts of the vocab to generate our training set.




###############################################################################
#                                                                             #
#                           Bag of Words with TF-IDF                          #
#                                                                             #
###############################################################################

# To utilize the TF-IDF scoring metric, we need multiple documents.  So first,
# we must get a corpus with multiple documents that are related.

import nltk
import sklearn

import sklearn.feature_extraction.text.TfidfTransformer
import sklearn.feature_extraction.text.TfidfVectorizer

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

