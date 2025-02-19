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

del reviews  # Save some memory

# TODO: Get all the words as the initial vocab
reviews_vocab = []

for review in reviews_df['reviews']:
    for word in nltk.word_tokenize(review):
        reviews_vocab.append(word)
        
# Get unique vocab
reviews_vocab_clean = list(set(reviews_vocab))
del reviews_vocab # Save some memory
        
# Remove stop words from vocab
reviews_vocab_clean = [token for token in reviews_vocab_clean if token not in nltk.corpus.stopwords.words('english')]

# Remove punctuation and small words from vocab
reviews_vocab_clean = [token for token in reviews_vocab_clean if len(token) > 1]

# Could also lowercase all words, but sometimes capitalization is important

# TODO: Looking through the data, we can see that some groups of words were 
# kept as one, separated by / (i.e. magician/inventor).  For the purposes of 
# this example, I have decided to split these into separate features.  
#
# Some of the words also contain an erroneous period that we will remove.
reviews_vocab_clean_split = []

for token in reviews_vocab_clean:
    # Remove any erroneous periods
    token = token.replace(".","")
    
    # Split up any multi-word tokens separated by "/"
    if token.find("/") != -1:
        reviews_vocab_clean_split.extend(token.split("/"))
    else:
        reviews_vocab_clean_split.append(token)

# We may now have duplicates again, so let's make the vocab set unique one
# more time.  Note this can all be combined in the for loop that initially gets
# the tokens for a more efficient script.
reviews_vocab_clean_split = list(set(reviews_vocab_clean_split))

del reviews_vocab_clean # Save some memory

# TODO: Go through each document (the 'reviews' column in the reviews_df) and
# get a count of each word.  This needs to be stored in an appropriate data
# structure.  For this example, we will use a DataFrame to hold all the 
# features, utilizing a dictionary to add new rows to each DataFrame
review_vocab_counts = []
for review in reviews_df["reviews"]:
    # Make sure to split multi-word tokens so they will match our vocab
    # Also remove erroneous periods.  The other tokens and stop words just 
    # won't find any matches in our DataFrame features, and thus will not need
    # to be removed here.
    token_count_dict = {}
    review = review.replace("/"," ")
    for review_token in nltk.word_tokenize(review):
        review_token = review_token.replace(".", "")
        
        if review_token in token_count_dict.keys():
            token_count_dict[review_token] += 1
        else:
            token_count_dict[review_token] = 1
            
    review_vocab_counts.append(token_count_dict)

# Due to memory issues, only use the first 1000 reviews
train_df = pd.DataFrame(data=review_vocab_counts[:1000], columns=reviews_vocab_clean_split)

# Where there are no matches, NaN will be placed.  These need to be numbers.
train_df.fillna(0, inplace=True)

# Taking a look at the train_df, we can see that this is a pretty sparse matrix

# Let's test to see if it was filled in correctly by checking the words in
# the first review against the columns of those tokens.
for token in review_vocab_counts[0]:
    review_vocab_count_val = review_vocab_counts[0][token]
    train_df_value = train_df.loc[0][token]
    print("token:", token, "   review token count:", review_vocab_count_val, "   train_df count:", train_df_value)

# Looks like we get a key error.  Why might this be?  Hint: check the stop words
# So, to fix this, we should be checking the cleaned vocab list when we are 
# getting our counts.







# # Sure is a lot of work.  Wouldn't it be nice if there was a library that just
# # did this for us?
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# vectorizer.fit_transform(reviews_df["reviews"])

# Done, noting that this is a sparse matrix numpy data structure

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

