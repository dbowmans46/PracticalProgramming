#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:22:26 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBreastCancerData
from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

###############################################################################
#                                                                             #
#                             Bagging Classifier                              #
#                                                                             #
###############################################################################

data_train, data_test, target_train, target_test = GetTrainTestSplitBreastCancerData()

# We will start with a classifier.  By setting the bootstrap to true, we are 
# using a bagging ensemble.  The n_jobs keyword sets the number of processors
# sckit-learn can use.  -1 means use as many as needed.  We will also use
# the out-of-bag instances for testing
bagger_model = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), 
                                 n_estimators=300, max_samples=75, bootstrap=True, 
                                 n_jobs=-1, oob_score=True)
bagger_model.fit(data_train, target_train)

# We can predict targets for the test data, and score as we did before
bagger_score = bagger_model.score(data_test, target_test)
print("Bagging Classifier Score: ", bagger_score)
print("Bagging Classifier OOB Score: ", bagger_model.oob_score_)

# Remember that we have not randomized the data over multiple tests, like using
# cross vaidation, so the scores may be somewhat different and not represent
# a good accuracy score for the model.

# Let's compare that with a single KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(data_train, target_train)
knn_score = knn_model.score(data_test, target_test)
print("KNN Classifier Score: ", knn_score,"\n\n\n")


###############################################################################
#                                                                             #
#                              Pasting Regressor                              #
#                                                                             #
###############################################################################

data_train, data_test, target_train, target_test = GetTrainTestSplitCAHousingData()

# For the pasting ensemble, we will use a regressor.  To creeate a pasting
# ensemble instance, we set the bootstrap keyword to false.  
paster_model = BaggingRegressor(LinearRegression(), n_estimators=300, 
                                max_samples=5000, bootstrap=False, n_jobs=-1)
paster_model.fit(data_train, target_train)
paster_score = paster_model.score(data_test, target_test)
print("Pasting Classifier Score: ", paster_score)

LinearRegression
# Let's compare that with a single linear regressor
lr_model = LinearRegression()
lr_model.fit(data_train, target_train)
lr_score = lr_model.score(data_test, target_test)
print("Linear Regressor Score: ", lr_score)