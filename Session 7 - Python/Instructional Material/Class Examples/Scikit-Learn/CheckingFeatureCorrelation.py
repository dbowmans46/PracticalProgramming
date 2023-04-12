#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn

# Pandas dataframe correlation matrix, same as Pearson's R correlation factor
# Only useful for linear correlations
import pandas as pd

# data_file_path = "../../In-Class Exercises/Data/housing.csv"
data_file_path = "../../In-Class Exercises/Data/housing NaN Strings.csv"
original_data_set = pd.read_csv(data_file_path)
print("Correlation matrix: ")
print(original_data_set.corr())

# We can also check the correlation of a single feature with all others
print("Correlations of just median_house_value:")
print(original_data_set.corr()["median_house_value"])
print("\n\n")


# This is Pearson's R correlation factor, same as the DataFrame.corr() method.
# Only useful for linear correlations
import scipy
pearson_r, p_val = scipy.stats.pearsonr(original_data_set["median_house_value"], original_data_set["latitude"])
print("pearson_r:", pearson_r)
print("p_val:", p_val,"\n\n")

# This is Spearman's R correlation factor, and can be used for non-linear
# correlations
spearman_r, p_val = scipy.stats.spearmanr(original_data_set["median_house_value"], original_data_set["latitude"])
print("spearman_r:", spearman_r)
print("p_val:", p_val,"\n\n\n")


# In addition to correlation factors, feature importances can tell us how
# much a feature affects targets.  For this example, we will also need to
# convert string data to numerical, imputemissing values, and manually split
# off the data and targets from a DataFrame

# First, convert string categories to numbers
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
ocean_cat_lb = lb.fit_transform(original_data_set['ocean_proximity'])

# Transpose for easy addition to dataframe
ocean_cat_lb = ocean_cat_lb.transpose()

# Drop string categorical data
data_encoded = original_data_set.drop('ocean_proximity', axis=1)

# Add numerical categorical data
for category_index in range(len(lb.classes_)):
    category_name = lb.classes_[category_index]
    data_encoded[category_name] = ocean_cat_lb[category_index]

# Let's replace the invalid values.  We will use the Imputer estimator as we saw above
# to estimate these values.
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy="median")
imputed_data = si.fit(data_encoded)
filled_in_data = imputed_data.transform(data_encoded)

# Let's create a dataframe to house our new data, and give each column an
# appropriate name.
# Since we added new columns, need to incorporate those into the list of columns
# for the new dataframe
new_col_names = original_data_set.columns.tolist()
new_col_names.remove('ocean_proximity')
new_col_names.extend(lb.classes_)
filled_in_df = pd.DataFrame(data=filled_in_data, columns=new_col_names)

# Set up the targets and the data
targets = filled_in_df["median_house_value"]
data =  filled_in_df.drop(labels="median_house_value", axis=1)

data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(data,
                                          targets,
                                          random_state=0)

# Train a model that can utilize feature_importances_ (any of the decision tree
# regressors will work)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(data_train, target_train)

# And finally, we can check the feature importance
col_names = filled_in_df.columns.tolist()
col_names.remove("median_house_value")
print("Feature Importances:")
for index in range(len(dtr.feature_importances_)):
    # Create an index holder.  Will be sliced to align text after index.
    index_str = str(index) + ")   "

    # Used to align feature importance values
    spacer_string = "                          "
    len_to_strip = len('housing_median_age') - len(col_names[index])
    print(index_str[:5], col_names[index], spacer_string[:len_to_strip], dtr.feature_importances_[index])