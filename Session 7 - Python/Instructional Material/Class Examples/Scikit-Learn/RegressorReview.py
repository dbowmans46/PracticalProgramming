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
import pandas as pd
from sklearn import model_selection

###############################################################################
#                                 Gather Data                                 #
###############################################################################

# from LoadScikitLearnDataSets import GetTrainTestSplitCAHousingData
# data_train, data_test, target_train, target_test = GetTrainTestSplitCAHousingData()

data_file_path = "../../In-Class Exercises/Data/NOAA Coastal Economic Data/ENOW_Industries/ENOW_Industries_2005_2020.csv"
data_df = pd.read_csv(data_file_path)


###############################################################################
#                               Data Preparation                              #
###############################################################################

# For this example, we want to be able to predict the employment values for
# jobs within coastal markets. NOAA data was obtained for coastal states, 
# including states on the great lakes.  We only care about data at the state
# level.
# Looking at the dataset, we see these columns:    
#     GeoID          - Derived from the 5-digit Federal Information Processing Standards codes
#     GeoName        - Name of the GeoScale (I.e. if GeoScale is state, this will be Maine, Washington, etc.)
#     GeoScale       - Scale the data applies to.  Contains a value of National or State.
#     Year
#     OceanSector    - Economic sector the data applies to
#     OceanSect_ID   - ID given to the sector
#     OceanIndustry  - The industry within the sector
#     OceanInd_ID    - ID given to the industry
#     Establishments - Number of businesses categorized in this group
#     Employment     - Number of employees
#     Wages          - Amount paid to workers
#     GDP            - Gross Domestic Product for the year
#     RealGDP        - GDP Normalized to year 2012 values



# First thing we will do is remove columns that duplicate data, or will not be
# helpful.  We will remove RealGDP since this data is scaled from GDP.  The 
# OCeanSect_ID and OceanInd_ID's will also be removed.  New values will be 
# created when we label binarize the OceanSector and OceanIndustry columns.  We
# do not want to just use the ID columns, since these may give the machine
# learning model inaccurate information on how correlated the ID's are.  
data_df.drop(['OceanSect_ID', 'OceanInd_ID', 'RealGDP'], axis=1, inplace=True)

# Since we only care about data at the state level, we can drop the rows where
# the GeoScale value is National, then drop the GeoScale column.
data_df.drop(data_df[data_df['GeoScale']=='National'].index, axis=0, inplace=True)
data_df.drop(['GeoScale'], axis=1, inplace=True)

# Values of -9999 in the Establishments, Employment, Wages, GDP, and RealGDP
# represent missing data.  These will be removed, since this will add inaccurate
# weights to the regressor we choose.
indices_to_drop = data_df[( (data_df['Establishments'] == -9999) | (data_df['Employment'] == -9999) | (data_df['Wages'] == -9999) | (data_df['GDP'] == -9999) )].index
data_df.drop(indices_to_drop, axis=0, inplace=True)

# Let's do a spot check to make sure we filtered correctly
print("Shape of data_df GDP where the value is -9999:", data_df[data_df['GDP'] == -9999].shape)


# Next step is to label binarize our GeoName, sector, and industry information
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
for col_name in ['GeoName', 'OceanSector', 'OceanIndustry']:
    temp_col_array = lb.fit_transform(data_df[col_name])
    temp_df = pd.DataFrame(data=temp_col_array, columns=lb.classes_)
    data_df = pd.concat([data_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
    data_df = data_df.drop(col_name, axis=1)

# Let's do a quick check to make sure the new columns exist as expected
print(data_df.columns)

# Finally, let's get our testing and training data
data_points_df = data_df.drop(['Employment'], axis=1, inplace=False)
targets_df = list(data_df['Employment'])
data_train, data_test, target_train, target_test = \
    model_selection.train_test_split(data_points_df, targets_df, random_state=0)

###############################################################################
#                              Regressor Selection                            #
###############################################################################
    
# from sklearn.neighbors import KNeighborsRegressor
# neighbor_count = 3
# print("K-Nearest Neighbors with", neighbor_count, "Neighbors")
# knn_model = KNeighborsRegressor(n_neighbors=neighbor_count, )
# knn_model.fit(data_train, target_train)
# MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, \
#                             target_train, target_test, 8)
# print("\n\n\n")


# from sklearn.tree import DecisionTreeRegressor
# dec_tree_reg = DecisionTreeRegressor(max_depth=5)
# dec_tree_reg.fit(data_train, target_train)
# MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
#                             target_train, target_test, 6)
# print("\n\n\n")


# from sklearn.linear_model import LinearRegression
# print("Linear Regression")
# l_reg_model = LinearRegression()
# l_reg_model.fit(data_train, target_train)
# MLHelper.FitAndGetAccuracy(l_reg_model, data_train, data_test, \
#                             target_train, target_test, 8)
# print("\n\n\n")


# from sklearn.linear_model import Ridge
# alpha_ridge = 5
# print("Ridge Regression with Alpha of", alpha_ridge)
# ridge_model = Ridge(alpha=alpha_ridge)
# MLHelper.FitAndGetAccuracy(ridge_model, data_train, data_test, \
#                             target_train, target_test, 8)
# print("\n\n\n")


# from sklearn.linear_model import Lasso

# alpha_lasso = 0.1
# print("Ridge Regression with Alpha of", alpha_lasso)
# lasso_model = Lasso(alpha=alpha_lasso)
# MLHelper.FitAndGetAccuracy(lasso_model, data_train, data_test, \
#                                 target_train, target_test, 8)
# print("\n\n\n")


###############################################################################
#                                   Metrics                                   #
###############################################################################

# # For usage of built-in metric functions, see the MLHelper.py file

# # Calculate the coefficient of determination

# # First, need to calculate the sum of squared residuals (SSR) of the 
# # predictions vs the actual target values from the test data.
# predictions = l_reg_model.predict(data_test)
# SSR = 0
# for index in range(len(predictions)):
#     SSR += (target_test[index] - predictions[index])**2

# # Next, we need to get the total sum of squares (TSS), calculated as the 
# # differences between the actual targets and the mean of the actual targets
# import numpy as np      
# target_test_mean_val = np.mean(target_test)
# TSS = 0
# for index in range(len(target_test)):
#     TSS += (target_test[index] - target_test_mean_val)**2
    
# coefficient_of_det = 1 - SSR/TSS
# print("Manually-calculated Coefficient of Determination (R^2):", coefficient_of_det)

# # MSE is the sum of squared residuals divided by the number of samples.  The 
# # squared sum of residuals have already been calculated for the coefficient
# # of determination
# manual_mse = SSR/len(predictions)
# print("Manually-calculated Mean Squared Error:", manual_mse)

# manual_rmse = manual_mse**(1/2)
# print("Manually-calculated Root Mean Squared Error:", manual_rmse)