#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:28:43 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBreastCancerData
from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn import model_selection

data_train, data_test, target_train, target_test = GetTrainTestSplitBreastCancerData()

# Hyperparameters used here:
#     max_depth - Number of decision boundaries to use for each tree
#     learning_rate - contribution of each tree to the ensemble's learning
#     subsmaple - Fraction of total instances each tree is trained to
gbt_model = GradientBoostingClassifier(max_depth=2, learning_rate=0.1, subsample=0.27)
gbt_model.fit(data_train, target_train)
gbt_model_score = gbt_model.score(data_test, target_test)
print("Gradient Boosted Tree Classifier Score: ", gbt_model_score)

# # We can iterate through to see the effect of different learning rate values
# print("Learning Rate       Score")
# print("-------------       -----")
# for learning_rate_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     gbt_model = GradientBoostingClassifier(max_depth=2, learning_rate=learning_rate_val, subsample=0.15)
#     gbt_model.fit(data_train, target_train)
#     gbt_model_score = gbt_model.score(data_test, target_test)
#     print("   ", learning_rate_val, "           ", gbt_model_score)

# # We can iterate through to see the effect of different subsample values
# print("Subsample           Score")
# print("---------           -----")
# for subsample_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     gbt_model = GradientBoostingClassifier(max_depth=2, learning_rate=0.3, subsample=subsample_val)
#     gbt_model.fit(data_train, target_train)
#     gbt_model_score = gbt_model.score(data_test, target_test)
#     print("   ", subsample_val, "           ", gbt_model_score)
