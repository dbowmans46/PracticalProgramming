#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:02:47 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData

# Setup our data
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data_train, data_test, target_train, target_test = GetTrainTestSplitCAHousingData()

# Don't forget to scale the data as needed.  Here, we are not applying a
# scalar to simplify the code, but scaling the data may be important 
# depending on the model you choose and the data you are processing.
dec_tree_model = DecisionTreeRegressor(max_depth=3, random_state=0)
dec_tree_model.fit(data_train, target_train)
y_pred_train = dec_tree_model.predict(data_train)
y_pred_test = dec_tree_model.predict(data_test)

# Mean squared error
from sklearn.metrics import mean_squared_error
mse_score_train = mean_squared_error(y_pred_train, target_train)
mse_score_test = mean_squared_error(y_pred_test, target_test)
print("MSE values for max_depth=3:")
print("mse_score_train:", mse_score_train)
print("mse_score_test:", mse_score_test,"\n")

# Coefficient of Determination (R^2)
from sklearn.metrics import r2_score
coeff_of_det_score_train = r2_score(y_pred_train, target_train)
coeff_of_det_score_test = r2_score(y_pred_test, target_test)
print("R^2 values for max_depth=3:")
print("coeff_of_det_score_train:", coeff_of_det_score_train)
print("coeff_of_det_score_test:", coeff_of_det_score_test,"\n\n\n")

# As we can see fro the MSE and R^2 values, this model using a max depth of 3
# is pretty attrocious.  We can change the max depth to get better results, 
# as needed.
best_test_coeff_det = {"max_depth":0, "R^2":0}
for max_depth_val in range(1,30):
    dec_tree_model = DecisionTreeRegressor(max_depth=max_depth_val, random_state=0)
    dec_tree_model.fit(data_train, target_train)
    y_pred_train = dec_tree_model.predict(data_train)
    y_pred_test = dec_tree_model.predict(data_test)
    
    mse_score_train = mean_squared_error(y_pred_train, target_train)
    mse_score_test = mean_squared_error(y_pred_test, target_test)
    print("MSE values for max_depth =",max_depth_val)
    print("mse_score_train:", mse_score_train)
    print("mse_score_test:", mse_score_test,"\n")
    
    coeff_of_det_score_train = r2_score(y_pred_train, target_train)
    coeff_of_det_score_test = r2_score(y_pred_test, target_test)
    print("R^2 values for max_depth = ", max_depth_val)
    print("coeff_of_det_score_train:", coeff_of_det_score_train)
    print("coeff_of_det_score_test:", coeff_of_det_score_test,"\n\n\n")
    
    if (coeff_of_det_score_test > best_test_coeff_det["R^2"]):
        best_test_coeff_det["max_depth"] = max_depth_val
        best_test_coeff_det["R^2"] = coeff_of_det_score_test

print("Highest R^2 test value is", best_test_coeff_det["R^2"], " at iteration", best_test_coeff_det["max_depth"])

# Prediction error plot
from sklearn.metrics import PredictionErrorDisplay

display = PredictionErrorDisplay.from_predictions(y_true=target_train, y_pred=y_pred_train, kind="actual_vs_predicted")
plt.show()

# There is also a method to use the PredictionErrorDisplay using the estimator
display = PredictionErrorDisplay.from_estimator(dec_tree_model, data_train, target_train, kind="actual_vs_predicted")
plt.title("Using from_estimator()")
plt.show()

# Residuals plot uses the same PredictionErrorDisplay function, just with
# a different input
display = PredictionErrorDisplay.from_predictions(y_true=target_train, y_pred=y_pred_train, kind="residual_vs_predicted")
plt.show()