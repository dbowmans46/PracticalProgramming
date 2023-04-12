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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

C_val = 1000
max_iterations = 100000

print("Accuracy without scaling:\n","--------------------\n")
log_reg_model = LogisticRegression(C=C_val, max_iter=max_iterations)
MLHelper.FitAndGetAccuracy(log_reg_model, data_train, data_test, \
                            target_train, target_test, 8)


print("\n\n\n")
print("Accuracy with scaling:\n","--------------------\n")

scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(data_train, target_train)
scaled_test_data = scaler.fit_transform(data_test, target_test)
log_reg_model = LogisticRegression(C=C_val, max_iter=max_iterations)
MLHelper.FitAndGetAccuracy(log_reg_model, scaled_train_data, scaled_test_data, \
                            target_train, target_test, 8)

print("\n\n\n")
print("Accuracy with scaling and cross validation:\n","--------------------\n")

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data['data'], data['target'])
cvs = cross_val_score(log_reg_model, scaled_data, data["target"], scoring="accuracy")

print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())


# Scaling using a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
print("\n\n\n")
print("Accuracy with Scaling and Using Pipelines:")

pipeline_w_scaler = Pipeline([
    ("scalar", StandardScaler()),
    ("log_reg", LogisticRegression(C=C_val, max_iter=max_iterations))
    ])
MLHelper.FitAndGetAccuracy(pipeline_w_scaler, data_train, data_test, \
                            target_train, target_test, 8)
cvs = cross_val_score(pipeline_w_scaler, data["data"], data["target"], scoring="accuracy")
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())