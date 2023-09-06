#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:22:55 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitMakeMoonsData

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

# Load the data
data_train, data_test, target_train, target_test = GetTrainTestSplitMakeMoonsData()

# Declare all of our models
lr_model = LogisticRegression(max_iter=100)
dtc_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
# Seting the probability=True for the SVC trainer allows us to utilize soft 
# voting.
svc_model = SVC(probability=True) 

estimators_list = [('lr', lr_model),
                   ('dtc', dtc_model),
                   ('knn', knn_model),               
                   ('svc', svc_model)
                   ]

# We can chose either hard voting or soft voting.  Hard voting does a simple
# vote between the different models, while soft voting takes into account
# the confidence of the data.
voting_model = VotingClassifier(estimators = estimators_list, voting='hard')
#voting_model = VotingClassifier(estimators = estimators_list, voting='soft')

# Let's check the accuracy score of each model individually and of the ensemble
from sklearn.metrics import accuracy_score
print("   Model                       Accuracy")
print("   -----                       --------")
for model in (lr_model, dtc_model, svc_model, voting_model):
    model.fit(data_train, target_train)
    model_predictions = model.predict(data_test)
    model_name = model.__class__.__name__
    
    # Format the output so the model and accurcy values are aligned under their 
    # headers
    print("   {0:<27} {1}".format(model_name, accuracy_score(target_test, model_predictions)))
    
# We can see that the voting ensemble does not always give us a better result
# than using a single model.  Also, becuase the data set we are using has
# random noise for each run, this will cause the voting classifier to be more
# accurate on some runs, and less accurate on others.  
