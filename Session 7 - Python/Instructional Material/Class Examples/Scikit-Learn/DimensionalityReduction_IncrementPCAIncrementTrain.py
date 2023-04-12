#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:09:09 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import sklearn
from MLHelper import MLHelper
from sklearn.decomposition import IncrementalPCA
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_data = iris_dataset['data']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_data = sc.fit_transform(iris_data)

num_batches = 10
# We cannot use a minimum variance input for incremental PCA, as we could with
# the normal PCA transformer.  Here, we must specify how many components we
# want.
component_count = iris_data.shape[1] - 2
incremental_pca = IncrementalPCA(n_components = component_count)

# We will use NumPy to split the data into equal batches, then feed the
# incremental PCA transformer
for batch_of_data in np.array_split(scaled_data, num_batches):
    incremental_pca.partial_fit(batch_of_data)
    data_reduced = incremental_pca.transform(scaled_data)
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data_reduced,
                                                  iris_dataset['target'],
                                                  random_state=0)

    # If we also need to save size in the output data, we can use an
    # incremental ML model to partially fit the data
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier()
    model.partial_fit(data_train, target_train, classes=(0,1,2))