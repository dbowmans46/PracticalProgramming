#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:01:13 2024

Copyright 2024 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
import pandas
from sklearn import model_selection

###############################################################################
#                                 Gather Data                                 #
###############################################################################




###############################################################################
#                               Data Preparation                              #
###############################################################################




###############################################################################
#                              Regressor Selection                            #
###############################################################################
    
from sklearn.neighbors import KNeighborsRegressor
neighbor_count = 3
print("K-Nearest Neighbors with", neighbor_count, "Neighbors")
knn_model = KNeighborsRegressor(n_neighbors=neighbor_count)
knn_model.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n\n")


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(max_depth=5)
dec_tree_reg.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
                            target_train, target_test, 6)
print("\n\n\n")


from sklearn.linear_model import LinearRegression
print("Linear Regression")
l_reg_model = LinearRegression()
l_reg_model.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(l_reg_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n\n")


from sklearn.linear_model import Ridge
alpha_ridge = 0.005
print("Ridge Regression with Alpha of", alpha_ridge)
ridge_model = Ridge(alpha=alpha_ridge)
MLHelper.FitAndGetAccuracy(ridge_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n\n")


from sklearn.linear_model import Lasso

alpha_lasso = 0.005
print("Ridge Regression with Alpha of", alpha_lasso)
lasso_model = Lasso(alpha=alpha_lasso)
MLHelper.FitAndGetAccuracy(lasso_model, data_train, data_test, \
                                target_train, target_test, 8)
print("\n\n\n")


###############################################################################
#                                   Metrics                                   #
###############################################################################

from sklearn.metrics import mean_squared_error