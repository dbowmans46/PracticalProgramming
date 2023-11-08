#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:16:14 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
import pandas
from sklearn import model_selection

###############################################################################
#                                 Gather Data                                 #
###############################################################################

# We will use the fire alarm inspection data to try and predict
# which ID's have had a recent annual inspection.
filepath = "../../In-Class Exercises/Data/Detroit Fire Alarm Inspection/Fire_Inspections.csv"
data = pandas.read_csv(filepath)
data = data.set_index('IO_ID')

###############################################################################
#                               Data Preparation                              #
###############################################################################

# There are quite a few records with missing lat and lon values
# I am choosing to keep those in for now, since missing coordinates may be an
# indicator in which ID's do not have up-to-date fire alarm inspections.
# These values do, however, need to be converted to numbers
import numpy as np
data = data.replace(np.nan,0)

# We need to convert strings to numbers, or drop the columns


# Split the data into training data and test data

data_points = data.drop('InspWithinLastYear', axis=1)
targets = data['InspWithinLastYear']

data_train, data_test, target_train, target_test = \
    model_selection.train_test_split(data_points, targets, random_state=0)








###############################################################################
#                                 Data Scaling                                #
###############################################################################




###############################################################################
#                             Classifier Selection                            #
###############################################################################

# We will try a bunch of classifiers to see which one works best
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(data_train, target_train)         
target_predictions = knn_model.predict(data_test)

# from sklearn.tree import DecisionTreeClassifier
# dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)  # Set the classifier type
# dec_tree_model.fit(data_train, target_train)            # Train the model with data
# dec_target_predictions = dec_tree_model.predict(data_test)

# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(max_iter=1000000, c=0.005)
# lr_model.fit(data_train, target_train)
# target_predictions = lr_model.predict(data_test)

# from sklearn.svm import LinearSVC
# svc_model = LinearSVC(max_iter=1e7, C=0.05)
# svc_model.fit(data_train, target_train)
# target_predictions = svc_model.predict(data_test)

# from sklearn.svm import SVC
# nonlinear_svc_model = SVC(kernel="rbf", gamma=5, C=1)
# nonlinear_svc_model.fit(data_train, target_train)

# from sklearn.ensemble import VotingClassifier
# lr_model = LogisticRegression(max_iter=100)
# dtc_model = DecisionTreeClassifier()
# knn_model = KNeighborsClassifier()
# # Seting the probability=True for the SVC trainer allows us to utilize soft 
# # voting.
# svc_model = SVC(probability=True) 

# estimators_list = [('lr', lr_model),
#                    ('dtc', dtc_model),
#                    ('knn', knn_model),               
#                    ('svc', svc_model)]

# voting_model = VotingClassifier(estimators = estimators_list, voting='soft')
# voting_model.fit(data_train, target_train)

# from sklearn.ensemble import BaggingClassifier
# bagger_model = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), 
#                                  n_estimators=300, max_samples=75, bootstrap=True, 
#                                  n_jobs=-1, oob_score=True)
# bagger_model.fit(data_train, target_train)

# paster_model = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), 
#                                  n_estimators=300, max_samples=75, bootstrap=False, 
#                                  n_jobs=-1, oob_score=True)
# paster_model.fit(data_train, target_train)

# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(n_estimators=6000, max_leaf_nodes=15, n_jobs=-1)
# rf_model.fit(data_train, target_train)

# from sklearn.ensemble import AdaBoostClassifier
# ada_model = AdaBoostClassifier(dec_tree_model, 
#                                n_estimators=5, 
#                                algorithm="SAMME.R", 
#                                learning_rate = 0.5)
# ada_model.fit(data_train, target_train)

# from sklearn.ensemble import GradientBoostingClassifier
# gbt_model = GradientBoostingClassifier(max_depth=2, learning_rate=0.1, subsample=0.27)
# gbt_model.fit(data_train, target_train)

###############################################################################
#                                   Metrics                                   #
###############################################################################


