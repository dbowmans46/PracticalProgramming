#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitIrisData
from sklearn.linear_model import LogisticRegression

data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

# Default model uses a c value of 1
# We may need to increase the number of iterations the model uses to optimize
# the prediction function.
lr_model = LogisticRegression(max_iter=1000000)
lr_model.fit(data_train, target_train)
target_predictions = lr_model.predict(data_test)
print("Logistic Regression Accuracy")
print("----------------------------")
MLHelper.FitAndGetAccuracy(lr_model, data_train, data_test, \
                            target_train, target_test, 8)


c_vals = [0.001]
for x in range(10):
    c_vals.append(10*c_vals[-1])

# What happens when we play around with c
for c_val in c_vals:
    print("Logistic Regression with c =",c_val)
    lr_model = LogisticRegression(C=c_val, max_iter=100000)  # Note that C is capitalized
    lr_model.fit(data_train, target_train)
    print("Accuracy Score:", str(lr_model.score(data_test,target_test)*100) + "%")
    print("\n")