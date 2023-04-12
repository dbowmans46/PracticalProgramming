#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:14:16 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from sklearn import model_selection
from MLHelper import MLHelper
from sklearn.manifold import LocallyLinearEmbedding

# For this example, we will use a data set with fewer dimensions
from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_data = iris_dataset['data']

from sklearn.preprocessing import StandardScaler
st_sc = StandardScaler()
scaled_data = st_sc.fit_transform(iris_data)

# n_components represents the number of dimensions in the manifold,
# and n_neighbors is the closest neighbors we will use to estimate the weights
# and thus the projected data in the manifold.
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
lle_transformed_data = lle.fit_transform(iris_data)
print(lle_transformed_data)

data_train, data_test, target_train, target_test = \
    model_selection.train_test_split(lle_transformed_data,
                                              iris_dataset['target'],
                                              random_state=0)

# For this example, the KNN classifier is again used for no particular reason.
# The number of neighbors was chosen to maximize the test score, then the train
# score respectively.
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=4)
MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)