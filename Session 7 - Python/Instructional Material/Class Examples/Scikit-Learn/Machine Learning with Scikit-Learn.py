# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import pandas as pd
import sklearn

'''
The hello-world of machine learning - The iris data set prediction.  This data
set gives the petal length, petal width, sepal length, sepal width, and name 
for 3 different species of iris flowers.  This dataset is setup to predict
the name of an iris flower given the petal length, petal width, sepal length, 
and sepal width as features.
'''
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# We can get a description of the dataset
iris_dataset.DESCR

# We can get the features 
iris_dataset.feature_names

# We can look at the data
iris_dataset.data

# We can see what the name of the actual species is, and what values the data 
# represents.  The term 'target' is used in scikit-learn to represent the
# actual value the data represents.
iris_dataset.target_names
iris_dataset.target

# Let's setup our first model.  We will use a 75% training to 25% testing data
# ratio.  First, let's separate our training and testing data.  We use the
# train_test_split function to get x and y values (data points and accurate 
# outcomes) for our training data, and our testing data.  This function
# will shuffle the data, using a pseudorandom number generator that takes
# the random_state argument as the seed.
from sklearn import model_selection
data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(iris_dataset['data'], 
                                             iris_dataset['target'], 
                                             random_state=0)

# We can look at our data to see how it was split
print("data_train.shape:", data_train.shape)
print("data_test.shape:", data_test.shape)
print("target_train.shape:", target_train.shape)
print("target_test.shape:", target_test.shape)

# This example will use the k-NN linear classifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)  # Set the classifier type
knn_model.fit(data_train, target_train)          # Train the model with data

# We have now built the model, stored in teh knn_classifier variable.  But how
# do we know if it is accurate?  Let us check this now by creating a prediction
# from our test data, and seeing how well it matches our test targets.
target_predictions = knn_model.predict(data_test)
print("predictions:", target_predictions)
print("Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%")
