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

# Option 1 - Use PolynomialFeatures and LinearSVC to add polynomial features

from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Parameters:
#    degree - The order to increase each feature, recursively
#    include_bias - Toggle to include bias column, which contains the intercepts
poly_features = PolynomialFeatures(degree=3, include_bias=False)
poly_features_data = poly_features.fit_transform(data_train)
poly_svm = LinearSVC(loss="hinge", max_iter=1000000)
poly_svm.fit(poly_features_data, target_train)

# Can shorten the chain by using Pipelines.  Also, below is a brief peek at
# scaling data.  Scaling data becomes necessary when the features span vastly
# different orders of magnitude.  SVM's are particularly sensitive to
# scaling
from sklearn.pipeline import Pipeline
poly_svm = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
    ("scalar", StandardScaler()),
    ("svm_clf", LinearSVC(C=5, loss="hinge", max_iter=1000000))
    ])
poly_svm.fit(data_train, target_train)
print("Linear Support Vector Machine Accuracy")
print("----------------------------")
MLHelper.FitAndGetAccuracy(poly_svm, data_train, data_test,  \
                            target_train, target_test, 8)


# Option 2 - Use SVC with arguments to use the kernel trick
from sklearn.svm import SVC

# coef0 adjusts high-degree polynomial coefficients and low-degree polynomial
# coefficients
nonlinear_svc_model = SVC(kernel="poly", degree=3, coef0=1, max_iter=100000)
nonlinear_svc_model.fit(data_train, target_train)
print("Non-linear Polynomial SVM Accuracy")
print("----------------------------")
MLHelper.FitAndGetAccuracy(nonlinear_svc_model, data_train, data_test,  \
                            target_train, target_test, 8)


# Use SVC with a different kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# gamma is another regularization parameter.  Larger gamma tightens the kernel
# boundary around its respective class, while larger gamma values broaden the
# boundaries.
rbf_kernel_svc_model = Pipeline([
    ("scalar", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=1))
    ])
rbf_kernel_svc_model.fit(data_train, target_train)
print("Non-linear RBF Kernel SVM Accuracy")
print("----------------------------")
MLHelper.FitAndGetAccuracy(rbf_kernel_svc_model, data_train, data_test,  \
                            target_train, target_test, 8)