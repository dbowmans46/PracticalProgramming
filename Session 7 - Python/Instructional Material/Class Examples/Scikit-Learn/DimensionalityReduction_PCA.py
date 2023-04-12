#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:05:00 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from sklearn.decomposition import PCA
from MLHelper import MLHelper

# For this example, we will use a data set with fewer dimensions
from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_data = iris_dataset['data']

# Note that for this example, the data is all within the same order of magnitude,
# and thus we do not need to scale the data. If this is not the case, the data
# should first be scaled before passing it to the PCA transformer.  For
# sake of completeness, the data will be scaled.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled_data = sc.fit_transform(iris_data)

# For this basic example, we will just reduce the number of dimensions by 1
principal_component_count = iris_data.shape[1] - 1
pca = PCA(n_components = principal_component_count)
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

# We can get the original features by inversing the transform
pca_inverted_data = pca.inverse_transform(pca_transformed_data)

# We can also see how much of the data we have lost by checking how much data
# is preserved in our reduced model.  This information is held in the
# explained_variance_ratio_ member.  We can see that we still have about 99.5%
# of the total variance in the reduced data, so using the reduced feature set
# is a good tradeoff (remember, keeping at least 95% is ta good starting
# point).

print("Explained Variance Ratios: ", pca.explained_variance_ratio_)
print("Total variance captured in the reduced model: ", sum(pca.explained_variance_ratio_))


# We can also let the algorithm determine how many dimensions we should keep
# based on the variance we want to maintain
min_variance = 0.95

# The only difference is using the fraction of fariance as the input to the PCA
# argument
pca = PCA(n_components = min_variance)
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

print("\n\n\nSetting Variance Fraction")
print("-------------------------------")
print("Explained Variance Ratios: ", pca.explained_variance_ratio_)
print("Total variance captured in the reduced model: ", sum(pca.explained_variance_ratio_))