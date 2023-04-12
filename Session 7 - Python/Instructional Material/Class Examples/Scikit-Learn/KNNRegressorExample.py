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
from LoadScikitLearnDataSets import GetTrainTestSplitBostonHousingData

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data_train, data_test, target_train, target_test = GetTrainTestSplitBostonHousingData()

tuning_parameter_vals = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]

from sklearn.neighbors import KNeighborsRegressor
for neighbor_count in range(1,20):
    knn_model = KNeighborsRegressor(n_neighbors=neighbor_count)
    knn_model.fit(data_train, target_train)
    # knn_model = Pipeline([
    #     ("scalar", StandardScaler()),
    #     ("knnf", KNeighborsRegressor(n_neighbors=neighbor_count))
    #     ])
    # knn_model.fit(data_train, target_train)
    # print(neighbor_count,"Neighbors Training Accuracy Score:", str(knn_model.score(data_train,target_train)*100) + "%")
    # print(neighbor_count,"Neighbors Test Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%\n")
    
    MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)