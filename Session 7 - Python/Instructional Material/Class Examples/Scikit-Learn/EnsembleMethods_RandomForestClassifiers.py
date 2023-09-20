#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:01:15 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBreastCancerData
from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import sklearn
from sklearn import model_selection

###############################################################################
#                                                                             #
#                          Random Forest Classifier                           #
#                                                                             #
###############################################################################

data = load_breast_cancer()

data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(data["data"], data["target"], random_state=0)


# Random Forest Classifier has most of all the same keywords and hyperparameters
# as a decision tree and bagging classifier.
rf_model = RandomForestClassifier(n_estimators=600, max_leaf_nodes=15, n_jobs=-1)
rf_model.fit(data_train, target_train)

# We can predict targets for the test data, and score as we did before
rf_score = rf_model.score(data_test, target_test)
print("Random Forest Classifier Score: ", rf_score)

# Just like with a decision tree, we can check feature importances using
# random forest classifiers.
for feature, score in zip(data["feature_names"], rf_model.feature_importances_):
    print("Feature '" + str(feature) + "' importance:", score)
