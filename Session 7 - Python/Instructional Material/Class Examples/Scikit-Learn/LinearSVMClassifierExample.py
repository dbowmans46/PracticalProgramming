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

data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

from sklearn.svm import LinearSVC

for c_val in [0.001, 1, 10000]:
    svc_model = LinearSVC(max_iter=1e7, C=c_val)
    svc_model.fit(data_train, target_train)
    target_predictions = svc_model.predict(data_test)
    print("Linear Support Vector Machine Accuracy")
    print("----------------------------")
    MLHelper.FitAndGetAccuracy(svc_model, data_train, data_test,  \
                                target_train, target_test, 8)