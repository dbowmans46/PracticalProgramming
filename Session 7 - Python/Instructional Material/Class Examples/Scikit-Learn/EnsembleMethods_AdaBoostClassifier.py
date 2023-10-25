#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:47:05 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBreastCancerData
from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn import model_selection

###############################################################################
#                                                                             #
#                             AdaBoost Classifier                             #
#                                                                             #
###############################################################################

data_train, data_test, target_train, target_test = GetTrainTestSplitBreastCancerData()

# We will be using a decision tree for our model, so train one first. Any
# classifier can be used, however.  Only one type of trainer can be used.
dec_tree_model = DecisionTreeClassifier(max_depth=3)

# For AdaBoost, we will be using the SAMME.R algorithm.  This is a version of 
# SAMME built for trainers that have the predict_proba() method for predicting
# class probabilities.  The SAMME.R algorithm can perform better than just
# the SAMME algorithm, so we will utilize SAMME.R here.
#
# We can also adjust the learning rate, which affects how quickly the 
# weights are changed over each training instance.  
#
# The algorithm has two options:
#     SAMME   - discrete boosting algorithm
#     SAMME.R - SAMME real boost method, typically converges faster than SAMME
ada_model = AdaBoostClassifier(dec_tree_model, 
                               n_estimators=600, 
                               algorithm="SAMME.R", 
                               learning_rate = 0.5)
ada_model.fit(data_train, target_train)

# We can predict targets for the test data, and score as we did before
ada_model_score = ada_model.score(data_test, target_test)
print("AdaBoost Classifier Score: ", ada_model_score)


# # Let's see if we can maximize accuracy by adjusting the number of estimators
# estimator_counts = [50*x for x in range(13)]
# estimator_counts = estimator_counts[1:]  # Remove the first index value of 0
# scores = []

# for estimator_count in estimator_counts:
#     ada_model = AdaBoostClassifier(dec_tree_model, 
#                                    n_estimators=estimator_count, 
#                                    algorithm="SAMME.R", 
#                                    learning_rate = 0.5)
#     ada_model.fit(data_train, target_train)
#     scores.append(ada_model.score(data_test, target_test))
    
# # Check the scores for the best number of estimators
# print("\n\n\n")
# print("{0:<35}{1}".format("Estimator Count", "Score"))
# print("{0:<29}{1}".format("---------------", "-----------------"))
# for index, estimator_count in enumerate(estimator_counts):
#     print("      {0:<22} {1}".format(estimator_count, scores[index]))

