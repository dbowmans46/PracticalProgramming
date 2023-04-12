#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

# Load data that has category-based values
data_file_path = "../../In-Class Exercises/Data/housing.csv"
data_set = pd.read_csv(data_file_path)

le = LabelEncoder()
encoded_ocean_data = le.fit_transform(data_set['ocean_proximity'])

# Remove the string-based data and put in the encoded data
data_encoded = data_set.drop('ocean_proximity', axis=1)
data_encoded['ocean_proximity'] = encoded_ocean_data
print(data_encoded.head(10))


# One-hot encoding the data
# This will build off of the numerically-categorized data we produced above
ohe = OneHotEncoder()

# The fit_transform method expects a 2D matrix, so we need to convert the
# encoded data from a 1D array to a 2D array.
encoded_ocean_data_reshaped = encoded_ocean_data.reshape(-1,1)
ohe_ocean_data = ohe.fit_transform(encoded_ocean_data_reshaped)

# By default, one-hot encoding returns a SciPy sparse matrix to save memory.
# To add this to our dataframe, we can convert it to a normal array.
ohe_ocean_data_arr = ohe_ocean_data.toarray()

# # If we transpose the matrix, the data will be in a format easier to make
# # new columns with for our Pandas dataframe
ohe_ocean_data_arr = ohe_ocean_data_arr.transpose()

# Now we can make a column for each category name.  To check what each category
# number corresponds to, we can use the "classes_" property.  We can also
# use each label directly, as we will to later.
print("Label categories:", le.classes_)

# As before, we can remove the old string-category column
data_ohe_encoded = data_set.drop('ocean_proximity', axis=1)

# Now we can add the new columns for each category, and add the appropriate
# data.

for category_index in range(len(le.classes_)):
    category_name = le.classes_[category_index]
    print("Category to add to the dataframe: ", category_name)
    data_ohe_encoded[category_name] = ohe_ocean_data_arr[category_index]

print(data_ohe_encoded.head(1))


# For sparse matrices, you will need to adapt the code to convert the data
# for it to work with a dataframe.  There are functions that can import
# a sparse matrix into a dataframe directly.


# We can use the LabelBinarizer to accomplish the label encoding, and the
# one-hot encoding at the same time.  This will produce separate arrays
# that we will still need to manipulate into a form that a Pandas dataframe
# will like.
lb = LabelBinarizer()
ocean_cat_lb = lb.fit_transform(data_set['ocean_proximity'])

# We can check if this output is the same as our previous output after
# applying the label encoding and one-hot encoding sequentially
print("LabelBinarizer Check:", ocean_cat_lb == ohe_ocean_data)