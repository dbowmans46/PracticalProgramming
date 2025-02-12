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

# import nltk

# dracula_filepath = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Books/Dracula.txt"
# with open(dracula_filepath) as fileHandle:
#     dracula_string = fileHandle.read()
# dracula_tokens = nltk.word_tokenize(dracula_string)
# draculaText = nltk.Text(dracula_tokens)
# draculaFreqDist = draculaText.vocab()



###############################################################################
#                                                                             #
#                             IMDB Review Data Set                            #
#                                                                             #
###############################################################################

# For the methods in this section, we need to have multiple corpus elements.  For the
# remaining class lectures and examples, we will use the IMDB data set 
# from http://ai.stanford.edu/~amaas/data/sentiment/ or
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

# We will be utilizing the Stanford source to practice some basic python skills
# First, let us read in the files and associate data with each review

import os
ratings_root_dir = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Data/aclImdb_v1/aclImdb/"
pos_train_dir = ratings_root_dir + "train/pos/"
neg_train_dir = ratings_root_dir + "train/neg/"

# This list will contain tuple triples with (text,rating,pos=1/neg=0)
reviews = []
for file_name in os.listdir(pos_train_dir):
    full_file_path = pos_train_dir + file_name
    with open(full_file_path, 'r') as file_handle:
        review_text = file_handle.read()

    # File name has the form <id>_<rating_0-10>.txt
    rating = file_name.split("_")[1].split(".")[0]
    reviews.append((review_text, rating, 1))

for file_name in os.listdir(neg_train_dir):
    full_file_path = neg_train_dir + file_name
    with open(full_file_path, 'r') as file_handle:
        review_text = file_handle.read()

    # File name has the form <id>_<rating_0-10>.txt
    rating = file_name.split("_")[1].split(".")[0]
    reviews.append((review_text,rating,0))
    
# Since the first half of reviews are positive and the second half are negative,
# we should randomize the data so the machine learning model does not make
# erroneous links in the data.
from random import shuffle

shuffle(reviews)

# Now that we have a set of data, let's put it in a DataFrame

import pandas as pd
reviews_df = pd.DataFrame(data=reviews, columns=["reviews","rating","sentiment"])


###############################################################################
#                                                                             #
#                           Bag of Words with Count                           #
#                                                                             #
###############################################################################

import nltk
from sklearn.feature_extraction.text import CountVectorizer

# TODO: Load in the IMDB data into a dataframe
bag_of_words_df = reviews_df.copy()
# TODO: Get all the words as the initial vocab
reviews_vocab = []

for review in bag_of_words_df['reviews']:
    for word in nltk.word_tokenize(review):
        reviews_vocab.append(word)
        
# TODO: Remove stop words from vocab
reviews_vocab_clean = [token for token in reviews_vocab if token not in nltk.corpus.stopwords.words('english')]

# TODO: Remove punctuation and small words from vocab
reviews_vocab_clean = [token for token in reviews_vocab_clean if len(token) > 1]

# TODO: Get unique words in vocab, and add to DataFrame to get counts
reviews_vocab_clean = list(set(reviews_vocab_clean))




###############################################################################
#                                                                             #
#                           Bag of Words with TF-IDF                          #
#                                                                             #
###############################################################################

# To utilize the TF-IDF scoring metric, we need multiple documents.  So first,
# we must get a corpus with multiple documents that are related.

# import nltk
# import sklearn

# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Create TF-IDF features
# vectorizer = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1, 2))


# vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
# train_vectors = vectorizer.fit_transform(X_train)
# test_vectors = vectorizer.transform(X_test)
# print(train_vectors.shape, test_vectors.shape)

