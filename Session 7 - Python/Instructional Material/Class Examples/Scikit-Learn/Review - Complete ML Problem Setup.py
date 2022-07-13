#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:52:59 2022

@author: doug
"""

from MLHelper import MLHelper
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

# Load data that has categpry-based values
data_file_path = "../../In-Class Exercises/Data/housing.csv"
data_set = pd.read_csv(data_file_path)

# Step 0 - Look at the data and understand it.  Look for data types, missing
# data, what the columns represent, data size, etc.

# Convert string categories to numbers.  Can go the route of using the label
# encoder to generate numbers, then the one hot encoder to create separate
# columns for each category: either the data point matches the column or doesn't
lb = LabelBinarizer()
lb_ocean_data = lb.fit_transform(data_set['ocean_proximity'])

# The output is a 2-D array.  If using the label encoder and one-hot enocder, 
# the output will be a SciPy sparse matrix, and needs converted to a form 
# Scikit-Learn will understand.

# We transpose the data to make it easier to put the data into a DataFrame
ohe_ocean_data = lb_ocean_data.transpose()

# Since the categories have been created, we can remove the old string-category 
# column
data_encoded = data_set.drop('ocean_proximity', axis=1)

# Add new column data to the DataFrame
for category_index in range(len(lb.classes_)):
    category_name = lb.classes_[category_index]
    print("Category to add to the dataframe: ", category_name)
    data_encoded[category_name] = ohe_ocean_data[category_index]
    
# Some of the data has invalid datas for the machine learning algorithms
# such as NaN's, infiniti, etc).  Here, we will just remove those data points.
# Next up in the class, we will learn ways to estimate missing parameters.
data_encoded_cleaned = data_encoded.dropna(axis=0)
    
# At this point, we have data.  There are a few things we need to do, but the
# order on what we do next depends on how we wish to proceed.  If we want to 
# split the data with train_test_split, we can split the data, then scale the 
# data, then use a machine learning alorithm.  If we want to use cross 
# validation to eliminate variance, we need to scale and apply the machine
# learning algorithm first.  Either way, we need to separate out our target
# we wish to predict from our data.
target = data_encoded_cleaned['median_house_value']
data = data_encoded_cleaned.drop('median_house_value', axis=1)

# Using train_test_split
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(data, target, random_state=0)

scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(data_train, target_train)
scaled_test_data = scaler.fit_transform(data_test, target_test)
tuning_parameter_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]

# We will use LASSO for this example, and iterate to try and find the best 
# hyperparameter for accuracy
for alpha_val in tuning_parameter_vals:
    print("Lasso for Alpha=", alpha_val)
    #lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=alpha_val))
    lasso_model = Lasso(alpha=alpha_val, max_iter=10000)
    MLHelper.FitAndGetAccuracy(lasso_model, data_train, data_test, 
                               target_train, target_test, 8)

# Using cross validation
# For cross validation, all the trainers need to be setup ahead of time.  We 
# will use the same constructs as above, but will not iterate through hyper-
# parameters this time.  Alpha=5 converges fast, which is why it was chosen here.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data, target)
lasso_model = Lasso(alpha=5, max_iter=50000)

# scoring for a regressor is different than scoring for a classifier.  This 
# example is a regression problem, so we will use r2.
cvs = cross_val_score(lasso_model, scaled_data, target, scoring="r2")
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())


