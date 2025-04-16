#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 21:44:38 2025

@author: doug
"""

from random import shuffle
import os
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

# Import the review data
ratings_root_dir = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Data/aclImdb_v1/aclImdb/"
pos_train_dir = ratings_root_dir + "train/pos/"
neg_train_dir = ratings_root_dir + "train/neg/"

# This list will contain tuple triples with (text,rating,pos=1/neg=0)
reviews = []

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
shuffle(reviews)

import pandas as pd
reviews_df = pd.DataFrame(data=reviews, columns=["reviews","rating","sentiment"])

# Note that we cannot split the reviews_df to generate our train/test split 
# since the TfidfVectorizer will generate a unique set of tokens for each
# fit_transform operation.  If we split the training and testing data before
# hand, the trainind data and testing data will most likely have a different
# set of tokens.

# Create TF-IDF features
vectorizer = TfidfVectorizer()
reviews_vectors = vectorizer.fit_transform(reviews_df['reviews'])
train_data, test_data, train_targets, test_targets = \
    train_test_split(reviews_vectors, 
                     reviews_df["sentiment"], 
                     test_size=0.25, 
                     random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=100) 
knn_model.fit(train_data, train_targets)         
target_predictions = knn_model.predict(test_data)

# The number of correctly identified positive values
# True Positive / (True Positive + False Negative)
recall_score_val = recall_score(test_targets, target_predictions)
print("Recall:", recall_score_val)

# Accuracy of positive scores
# True Positives / (True Positives + False Positives)
precision_score_val = precision_score(test_targets, target_predictions)
print("Precision:", precision_score_val)

# Receiver operating characteristic area under the curve
# Generalized performance score.  The higher the curve fits to the top left of
# the graph, the better
roc_auc_score_val = roc_auc_score(test_targets, target_predictions)
print("ROC AUC:", roc_auc_score_val)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(test_targets, target_predictions)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix")
plt.show()

# Precision recall curve
prd = PrecisionRecallDisplay.from_predictions(test_targets, target_predictions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Graph Plotting from from_estimator() Method")
plt.show()

# ROC Curve
prd = RocCurveDisplay.from_predictions(test_targets, target_predictions)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph Plotting from from_estimator() Method")
plt.show()
