#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:11:02 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper

# This example is almost identical to the first with PCA, we are just going
# to pass an additional argument to the PCA transformer.

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_data = iris_dataset['data']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_data = sc.fit_transform(iris_data)

principal_component_count = iris_data.shape[1] - 2

# Here is where the only difference occurs, using a different solver with PCA.
pca = PCA(n_components = principal_component_count, svd_solver='randomized')
pca_transformed_data = pca.fit_transform(scaled_data)
print(pca_transformed_data)

data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(pca_transformed_data,
                                              iris_dataset['target'],
                                              random_state=0)

# For this example, the KNN classifier is used for no particular reason,
# and the number of neighbors chosen for no particular reason, as well.
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)