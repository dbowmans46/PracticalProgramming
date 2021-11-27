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

print("\nIntro Model - K-Nearest Neighbors Classifier\n")

# We can look at our data to see how it was split
print("data_train.shape:", data_train.shape)
print("data_test.shape:", data_test.shape)
print("target_train.shape:", target_train.shape)
print("target_test.shape:", target_test.shape)

# This example will use the k-NN linear classifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)  # Set the classifier type
knn_model.fit(data_train, target_train)          # Train the model with data

# We have now built the model, stored in the knn_classifier variable.  But how
# do we know if it is accurate?  Let us check this now by creating a prediction
# from our test data, and seeing how well it matches our test targets.
target_predictions = knn_model.predict(data_test)
print("predictions:", target_predictions)
print("Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%")




###############################################################################
#                                                                             #
#                    K-Nearest Neighbors Classifier                           #
#                                                                             #
###############################################################################

print("\n\n\n\nK-Nearest Neighbors Classifier\n")

# How does accuracy change with the number of neighbors?
for neighbors in range(1,11):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors) 
    knn_model.fit(data_train, target_train)         
    target_predictions = knn_model.predict(data_test)
    knn_score = round(knn_model.score(data_test,target_test)*100,8)
    print("K-Nearest Neighbors accuracy neighbors=" + str(neighbors) + " score:", str(knn_score) + "%")


print("\n\n\n\n")


###############################################################################
#                                                                             #
#                       Decision Tree Classifier                              #
#                                                                             #
###############################################################################

print("\nDecision Tree Classifier\n")


# This example will use the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)  # Set the classifier type
dec_tree_model.fit(data_train, target_train)            # Train the model with data
dec_target_predictions = dec_tree_model.predict(data_test)
print("Decision tree accuracy score:", str(dec_tree_model.score(data_test,target_test)*100) + "%")

# Viewing the decision tree
# Create a document that represents the tree
from sklearn.tree import export_graphviz
tree_file_name = "decision_tree.dot"
class_name_vals = iris_dataset.target_names
export_graphviz(dec_tree_model,out_file=tree_file_name, \
                class_names=iris_dataset.target_names, \
                feature_names=iris_dataset.feature_names, \
                impurity=False, filled=True)
    
# Load the document and view
import graphviz
with open(tree_file_name) as fileHandle:
    dot_graph = fileHandle.read()

graphviz.Source(dot_graph)
# If this does not work, you can always run dot.exe from the Graphviz
# installation, and generate the graph manually.


# How does accuracy change with the number of decisions?
for max_depth_val in range(1,11):
    dtm = DecisionTreeClassifier(max_depth=max_depth_val, random_state=0)  # Set the classifier type
    dtm.fit(data_train, target_train)            # Train the model with data
    dtm_target_predictions = dtm.predict(data_test)
    dtm_score = round(dtm.score(data_test,target_test)*100,8)
    print("Decision tree accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score) + "%")


# Why does the accuracy not improve after so many depth levels?  Hint: How many
# attributes do we have?


###############################################################################
#                                                                             #
#                     K-Nearest Neighbors Regressor                           #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                        Decision Tree Regressor                              #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                       Linear Regression Regressor                           #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                        Ridge Regression Regressor                           #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                        Lasso Regression Regressor                           #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                       Linear Regression Regressor                           #
#                                                                             #
###############################################################################







