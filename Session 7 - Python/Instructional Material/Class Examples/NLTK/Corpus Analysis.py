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
#ratings_root_dir = "D:/repo/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Data/aclImdb_v1/aclImdb/"
pos_train_dir = ratings_root_dir + "train/pos/"
neg_train_dir = ratings_root_dir + "train/neg/"

# This list will contain tuple triples with (text,rating,pos=1/neg=0)
reviews = []

# TODO: These for loops have the same # of iterations.  Run in paralle to
# possibly save time.
for file_name in os.listdir(pos_train_dir):
    full_file_path = pos_train_dir + file_name
    with open(full_file_path, 'r', encoding="UTF-8") as file_handle:
        review_text = file_handle.read()

    # File name has the form <id>_<rating_0-10>.txt
    rating = file_name.split("_")[1].split(".")[0]
    reviews.append((review_text, rating, 1))

for file_name in os.listdir(neg_train_dir):
    full_file_path = neg_train_dir + file_name
    with open(full_file_path, 'r', encoding="UTF-8") as file_handle:
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

# Because of memory constraints, we will limit the number of reviews for this
# tutorial
num_docs = 500
reviews_df = reviews_df.head(num_docs)


###############################################################################
#                                                                             #
#                           Bag of Words with Count                           #
#                                                                             #
###############################################################################

import nltk
import numpy as np

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
    # to be removed here.  We are going to add an extra column that holds
    # the total tokens within a document.  This will be needed later, and is
    # more efficient to calculate here
    token_count_dict = {"words_in_doc_count":0}
    review = review.replace("/"," ")
    for review_token in nltk.word_tokenize(review):
        token_count_dict["words_in_doc_count"] += 1
        review_token = review_token.replace(".", "")
        
        if review_token in token_count_dict.keys():
            token_count_dict[review_token] += 1
        else:
            token_count_dict[review_token] = 1
            
    review_vocab_counts.append(token_count_dict)

# Due to memory issues, only use the first 1000/25000 reviews
reviews_vocab_clean_split.append("words_in_doc_count") # Add the word count in this document to our list of vocab
train_df = pd.DataFrame(data=review_vocab_counts, columns=reviews_vocab_clean_split)

# Where there are no matches, NaN will be placed.  These need to be numbers.
train_df.fillna(0, inplace=True)

# Taking a look at the train_df, we can see that this is a pretty sparse matrix
# Convert to a sparse DataFrame to save some memory.  This will not store any 
# 0 values in the DataFrame in such a way that they can be repopulated later
# train_sparse_df = train_df.astype(pd.SparseDtype("int", 0))
# del train_df

# # We can check the compression as below
# print("Sparse DF Density:", train_sparse_df.sparse.density)

# Let's test to see if it was filled in correctly by checking the words in
# the first review against the columns of those tokens.
# for token in review_vocab_counts[0]:
#     review_vocab_count_val = review_vocab_counts[0][token]
#     train_df_value = train_sparse_df.loc[0][token]
#     print("token:", token, "   review token count:", review_vocab_count_val, "   train_df count:", train_df_value)

# Looks like we get a key error.  Why might this be?  Hint: check the stop words
# Note that this isn't necessarily an error, since some tokens have been removed.
# Another issue is the set() conversion does not preserve order, so we cannot
# simply check the values sequentially.  We can also only check words that
# should be in our feature set by not checking stop words and not checking
# punctuation.
# for token in review_vocab_counts[0]:
#     if len(token) > 1 and token not in nltk.corpus.stopwords.words('english'):
#         review_vocab_count_val = review_vocab_counts[0][token]
#         train_df_value = train_sparse_df.loc[0][token]
#         print("token:", token, "   review token count:", review_vocab_count_val, "   train_df count:", train_df_value)


# # Sure is a lot of work.  Wouldn't it be nice if there was a library that just
# # did this for us?
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# tokenized_data = vectorizer.fit_transform(reviews_df["reviews"])

# Done, noting that this is a sparse matrix numpy data structure

###############################################################################
#                                                                             #
#                           Bag of Words with TF-IDF                          #
#                                                                             #
###############################################################################

# We will build off our current data set since we need the count of tokens
# to calculate the TF-IDF.  Remember that we currently have token counts 
# per document in each row of our data.

import nltk
import sklearn

# First, we will calculate the value manually. 
# Get the TF value by getting the count of the token over the count of all
# tokens in this documents.  We will update the count values in place.
def generate_tf_vals(row, count_col_name = "words_in_doc_count"):
    for column in row.keys():
        if column != count_col_name:
            # Update the score for each token except the column housing the
            # total number of tokens in the document
            row[column] = row[column]/row[count_col_name]
    return row
    
train_df = train_df.apply(generate_tf_vals, axis=1)

 

# Get the number of documents this token exists in and calculate IDF val
# as ln (docs/docs where word appears).  We will create
# a dictionary for this to make the mapping easier.
import math
token_idf_vals = {}
for token in train_df.columns:
    count_of_docs_with_token = (train_df[token] != 0.0).sum()
    # In the math library, log is the natural log, ln
    token_idf_vals[token] = math.log(num_docs/count_of_docs_with_token)
    
# Just as above, we will apply a map to update the current TF values.  We could
# also simplify by making 1 function that generates the tf-idf value in one go.
def generate_tf_idf_vals(row_with_tf_vals, token_idf_vals, count_col_name = "words_in_doc_count"):
    for column in row_with_tf_vals.keys():
        if column != count_col_name:
            row_with_tf_vals[column] = row_with_tf_vals[column]*token_idf_vals[column]
    return row_with_tf_vals

# Apply the function
train_df = train_df.apply(lambda row: generate_tf_idf_vals(row, token_idf_vals), axis=1)

# Output the data to a CSV for future analysis without maxing memory usage
train_df.to_csv("tf-idf_data.csv")

# At this point, we now have a data frame that we can use to train a machine
# learning model of our choice.


# Just as with a simple count, there are already methods that can do this for
# us.
# sklearn.feature_extraction.text.TfidfTransformer can be utilized if you
# already have a matrix of token counts.
# sklearn.feature_extraction.text.TfidfVectorizer can be used if you are starting
# with raw documents, such as is the case with our example here.

# from sklearn.feature_extraction.text import TfidfVectorizer

# # Create TF-IDF features
# vectorizer = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1, 2))


# vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
# train_vectors = vectorizer.fit_transform(X_train)
# test_vectors = vectorizer.transform(X_test)
# print(train_vectors.shape, test_vectors.shape)




# So, if there are methods that already do this for us, why go through all the 
# hassle of doing it manually?  Sometimes, you need to customize the weighting
# or the algorithm to better represent the corpus you are working with.
