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
# helpful.  The GeoID data is stored in the GeoName column, and can be removed.
# We will remove RealGDP since this data is scaled from GDP.  The 
# OCeanSect_ID and OceanInd_ID's will also be removed.  New values will be 
# created when we label binarize the OceanSector and OceanIndustry columns.  We
# do not want to just use the ID columns, since these may give the machine
# learning model inaccurate information on how correlated the ID's are.  
data_df.drop(['GeoID', 'OceanSect_ID', 'OceanInd_ID', 'RealGDP'], axis=1, inplace=True)

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

# For some of the values in OceanSector and OceanIndustry, an '&' is used 
# instead of 'and' (or vic versa).  Change all the '&' characters to 'and' so
# these records do not seem like they are representing different identifiers.
# The .str member of the Series is how we treat Series member values as strings  
# in Pandas.
data_df['OceanSector'] = data_df['OceanSector'].str.replace('&', 'and')
data_df['OceanIndustry'] = data_df['OceanIndustry'].str.replace('&', 'and')

# We could also replace every character in a DataFrame by using the replace()
# method of the DataFrame, like below:
# data_df = data_df.replace('&', 'and')
#
# Notice with a Series (or DataFrame column), we need to look in the .str 
# member of the Series, while we can use .replace() directly with the DataFrame

# We have a similar problem with one of the industry labels, where on some
# records, 'Oil and Gas Exploration and Production' is truncated to 'Oil and 
# Gas Exploration and Product' or 'Oil & Gas Exploration and Productio'.  Since
# we have alread converted the '&' symbols to 'and', the 'Oil & Gas Exploration 
# and Productio' is now 'Oil and Gas Exploration and Productio'.
#
# Let's restore all of them to the full version, as this will be the clearest 
# label. Because the string 'Oil and Gas Exploration and Product' is a 
# substring of 'Oil and Gas Exploration and Production', if we just replace 
# 'Oil and Gas Exploration and Product' with 'Oil and Gas Exploration and 
# Production', the 'Oil and Gas Exploration and Production' values will be 
# converted to 'Oil and Gas Exploration and Productionion'.  Because of this, 
# we will first convert 'Oil and Gas Exploration and Production' to 'Oil and 
# Gas Exploration and Product', then convert 'Oil and Gas Exploration and 
# Product' to 'Oil and Gas Exploration and Production'.  The same problem 
# occurs with 'Oil and Gas Exploration and Productio', so we will convert it 
# first, as well.
data_df['OceanIndustry'] = \
    data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Production',
                                         'Oil and Gas Exploration and Product')
data_df['OceanIndustry'] = \
    data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Productio',
                                         'Oil and Gas Exploration and Product')
data_df['OceanIndustry'] = \
    data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Product',
                                         'Oil and Gas Exploration and Production')

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
    
from sklearn.neighbors import KNeighborsRegressor
neighbor_count = 3
print("K-Nearest Neighbors with", neighbor_count, "Neighbors")
knn_model = KNeighborsRegressor(n_neighbors=neighbor_count, )
knn_model.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n\n")


from sklearn.tree import DecisionTreeRegressor
print("Decision Tree")
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
alpha_ridge = 5
print("Ridge Regression with Alpha of", alpha_ridge)
ridge_model = Ridge(alpha=alpha_ridge)
MLHelper.FitAndGetAccuracy(ridge_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n\n")


from sklearn.linear_model import Lasso

alpha_lasso = 0.1
print("Ridge Regression with Alpha of", alpha_lasso)
lasso_model = Lasso(alpha=alpha_lasso)
MLHelper.FitAndGetAccuracy(lasso_model, data_train, data_test, \
                                target_train, target_test, 8)
print("\n\n\n")


###############################################################################
#                                   Metrics                                   #
###############################################################################

# For usage of built-in metric functions, see the MLHelper.py file

# Calculate the coefficient of determination

# First, need to calculate the sum of squared residuals (SSR) of the 
# predictions vs the actual target values from the test data.
predictions = ridge_model.predict(data_test)
SSR = 0
for index in range(len(predictions)):
    SSR += (target_test[index] - predictions[index])**2

# Next, we need to get the total sum of squares (TSS), calculated as the 
# differences between the actual targets and the mean of the actual targets
import numpy as np      
target_test_mean_val = np.mean(target_test)
TSS = 0
for index in range(len(target_test)):
    TSS += (target_test[index] - target_test_mean_val)**2
    
coefficient_of_det = 1 - SSR/TSS
print("Manually-calculated Coefficient of Determination (R^2):", coefficient_of_det)

# MSE is the sum of squared residuals divided by the number of samples.  The 
# squared sum of residuals have already been calculated for the coefficient
# of determination
manual_mse = SSR/len(predictions)
print("Manually-calculated Mean Squared Error:", manual_mse)

manual_rmse = manual_mse**(1/2)
print("Manually-calculated Root Mean Squared Error:", manual_rmse)


###############################################################################
#                              Making Predictions                             #
###############################################################################

new_data_file_path = "../../In-Class Exercises/Data/NOAA Coastal Economic Data/ENOW_Industries/ENOW_Industries_2021_Fake.csv"
new_data_df = pd.read_csv(new_data_file_path)

# We will need to do the same preprocessing to generate the same data format
# as our training data.  The new data does not contain the employment data, 
# as this is theoretically the data we are trying to predict.  The RealGDP has 
# also been removed.
new_data_df.drop(['GeoID', 'OceanSect_ID', 'OceanInd_ID'], axis=1, inplace=True)

new_data_df.drop(new_data_df[new_data_df['GeoScale']=='National'].index, axis=0, inplace=True)
new_data_df.drop(['GeoScale'], axis=1, inplace=True)
indices_to_drop = new_data_df[( (new_data_df['Establishments'] == -9999) | (new_data_df['Wages'] == -9999) | (new_data_df['GDP'] == -9999) )].index
new_data_df.drop(indices_to_drop, axis=0, inplace=True)
new_data_df['OceanSector'] = new_data_df['OceanSector'].str.replace('&', 'and')
new_data_df['OceanIndustry'] = new_data_df['OceanIndustry'].str.replace('&', 'and')
new_data_df['OceanIndustry'] = \
    new_data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Production',
                                             'Oil and Gas Exploration and Product')
new_data_df['OceanIndustry'] = \
    new_data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Productio',
                                             'Oil and Gas Exploration and Product')
new_data_df['OceanIndustry'] = \
    new_data_df['OceanIndustry'].str.replace('Oil and Gas Exploration and Product',
                                             'Oil and Gas Exploration and Production')

lb = LabelBinarizer()
for col_name in ['GeoName', 'OceanSector', 'OceanIndustry']:
    temp_col_array = lb.fit_transform(new_data_df[col_name])
    temp_df = pd.DataFrame(data=temp_col_array, columns=lb.classes_)
    new_data_df = pd.concat([new_data_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
    new_data_df = new_data_df.drop(col_name, axis=1)

# We can then use the predict() method of our already-trained models, using 
# the new data as inputs to get the predicted values.
knn_predictions = knn_model.predict(new_data_df)
dec_tree_predictions = dec_tree_reg.predict(new_data_df)
l_reg_predictions = l_reg_model.predict(new_data_df)
ridge_predictions = ridge_model.predict(new_data_df)
lasso_predictions = lasso_model.predict(new_data_df)

# These values all correspond respectively with the rows of our DataFrame.  We
# can add them back in to the new data, if we want all the data together.
new_data_and_predictions_df = new_data_df
new_data_and_predictions_df['knn_predictions'] = knn_predictions
new_data_and_predictions_df['dec_tree_predictions'] = dec_tree_predictions
new_data_and_predictions_df['l_reg_predictions'] = l_reg_predictions
new_data_and_predictions_df['ridge_predictions'] = ridge_predictions
new_data_and_predictions_df['lasso_predictions'] = lasso_predictions

# Notice that there are some negative values for predicitons.  These may have
# been created from the -9999 values (if any were not filtered), or possibly
# because we did not do any data scaling.




