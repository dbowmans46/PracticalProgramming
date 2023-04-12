#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitIrisData
from sklearn.datasets import load_iris

data = load_iris()
data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

# test_size + train_size <= 1.00
sklearn.model_selection.train_test_split(data['data'],
                                          data['target'],
                                          test_size=0.3,
                                          train_size=0.7,
                                          random_state=0)

from sklearn.linear_model import LogisticRegression
tts_model = LogisticRegression(max_iter=500000)
MLHelper.FitAndGetAccuracy(tts_model, data_train, data_test, \
                            target_train, target_test, 4)