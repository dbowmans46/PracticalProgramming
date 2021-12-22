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
from MLHelper import MLHelper


'''
The hello-world of machine learning - The iris data set prediction.  This data
set gives the petal length, petal width, sepal length, sepal width, and name 
for 3 different species of iris flowers.  This dataset is setup to predict
the name of an iris flower given the petal length, petal width, sepal length, 
and sepal width as features.
'''
# from sklearn.datasets import load_iris
# iris_dataset = load_iris()

# # We can get a description of the dataset
# iris_dataset.DESCR

# # We can get the features 
# iris_dataset.feature_names

# # We can look at the data
# iris_dataset.data

# # We can see what the name of the actual species is, and what values the data 
# # represents.  The term 'target' is used in scikit-learn to represent the
# # actual value the data represents.
# iris_dataset.target_names
# iris_dataset.target

# Let's setup our first model.  We will use a 75% training to 25% testing data
# ratio.  First, let's separate our training and testing data.  We use the
# train_test_split function to get x and y values (data points and accurate 
# outcomes) for our training data, and our testing data.  This function
# will shuffle the data, using a pseudorandom number generator that takes
# the random_state argument as the seed.
# from sklearn import model_selection
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(iris_dataset['data'], 
#                                               iris_dataset['target'], 
#                                               random_state=0)
    
# from sklearn.datasets import fetch_california_housing
# ca_housing_data = fetch_california_housing()
# x_train, x_test, y_train, y_test = \
#     sklearn.model_selection.train_test_split(ca_housing_data['data'], 
#                                           ca_housing_data['target'], 
#                                           random_state=0)
    
# from sklearn.datasets import load_boston
# boston_data = load_boston()
# x_train, x_test, y_train, y_test = \
#     sklearn.model_selection.train_test_split(boston_data['data'], 
#                                           boston_data['target'], 
#                                           random_state=0)
    
# from sklearn.datasets import load_diabetes
# diabetes_data = load_diabetes()
# x_train, x_test, y_train, y_test = \
#     sklearn.model_selection.train_test_split(diabetes_data['data'], 
#                                           diabetes_data['target'], 
#                                           random_state=0)
   
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()
data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(breast_cancer_data['data'], 
                                          breast_cancer_data['target'], 
                                          random_state=0)

# print("\nIntro Model - K-Nearest Neighbors Classifier\n")

# # We can look at our data to see how it was split
# print("data_train.shape:", data_train.shape)
# print("data_test.shape:", data_test.shape)
# print("target_train.shape:", target_train.shape)
# print("target_test.shape:", target_test.shape)

# # This example will use the k-NN linear classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=3)  # Set the classifier type
# knn_model.fit(data_train, target_train)          # Train the model with data

# # We have now built the model, stored in the knn_classifier variable.  But how
# # do we know if it is accurate?  Let us check this now by creating a prediction
# # from our test data, and seeing how well it matches our test targets.
# target_predictions = knn_model.predict(data_test)
# print("predictions:", target_predictions)
# print("Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%")




###############################################################################
#                                                                             #
#                    K-Nearest Neighbors Classifier                           #
#                                                                             #
###############################################################################

# print("\n\n\n\nK-Nearest Neighbors Classifier\n")
# from sklearn.neighbors import KNeighborsClassifier

# # How does accuracy change with the number of neighbors?
# for neighbors in range(1,11):
#     knn_model = KNeighborsClassifier(n_neighbors=neighbors) 
#     knn_model.fit(data_train, target_train)         
#     target_predictions = knn_model.predict(data_test)
#     knn_score = round(knn_model.score(data_test,target_test)*100,8)
#     print("K-Nearest Neighbors accuracy neighbors=" + str(neighbors) + " score:", str(knn_score) + "%")


# print("\n\n\n\n")


###############################################################################
#                                                                             #
#                       Decision Tree Classifier                              #
#                                                                             #
###############################################################################


# # This example will use the decision tree classifier
# from sklearn.tree import DecisionTreeClassifier

# dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)  # Set the classifier type
# dec_tree_model.fit(data_train, target_train)            # Train the model with data
# dec_target_predictions = dec_tree_model.predict(data_test)

# #print("Decision tree accuracy score:", str(dec_tree_model.score(data_test,target_test)*100) + "%")
# print("Decision Tree Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(dec_tree_model, data_train, data_test, target_train, target_test, 8)
# print("\n\n")

# # Viewing the decision tree
# # Create a document that represents the tree
# from sklearn.tree import export_graphviz
# tree_file_name = "decision_tree.dot"
# class_name_vals = iris_dataset.target_names
# export_graphviz(dec_tree_model,out_file=tree_file_name, \
#                 class_names=iris_dataset.target_names, \
#                 feature_names=iris_dataset.feature_names, \
#                 impurity=False, filled=True)
    
# # Load the document and view
# import graphviz
# with open(tree_file_name) as fileHandle:
#     dot_graph = fileHandle.read()

# graphviz.Source(dot_graph)
# # If this does not work, you can always run dot.exe from the Graphviz
# # installation, and generate the graph manually.


# # How does accuracy change with the number of decisions?
# for max_depth_val in range(1,11):
#     dtm = DecisionTreeClassifier(max_depth=max_depth_val, random_state=0)  # Set the classifier type
#     dtm.fit(data_train, target_train)            # Train the model with data
#     dtm_target_predictions = dtm.predict(data_test)
#     dtm_score = round(dtm.score(data_test,target_test)*100,8)
#     print("Decision tree accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score) + "%")


# # Why does the accuracy not improve after so many depth levels?  Hint: How many
# # attributes do we have?



###############################################################################
#                                                                             #
#                     Logistic Regression Classifier                          #
#                                                                             #
###############################################################################

# from sklearn.linear_model import LogisticRegression

# # Default model uses a c value of 1
# # We may need to increase the number of iterations the model uses to optimize
# # the prediction function.
# lr_model = LogisticRegression(max_iter=100000)
# lr_model.fit(data_train, target_train)
# target_predictions = lr_model.predict(data_test)
# # print("Logistic Regression Accuracy Score:", str(lr_model.score(data_test,target_test)*100) + "%")
# print("Logistic Regression Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(lr_model, data_train, data_test, target_train, target_test, 8)


# # What happens when we play around with c
# for c_val in [0.01, 0.1, 1, 10, 100]:
#     print("Logistic Regresion with c =",c_val)
#     lr_model = LogisticRegression(c=c_val)
#     lr_model.fit(data_train, target_train)
#     print("Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%")
#     print("\n")
    
    

###############################################################################
#                                                                             #
#                      Linear Support Vector Machine                          #
#                                                                             #
###############################################################################

# from sklearn.svm import LinearSVC

# svc_model = LinearSVC(max_iter=1000000)
# svc_model.fit(data_train, target_train)
# target_predictions = svc_model.predict(data_test)
# print("Linear Support Vector Machine Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(svc_model, data_train, data_test, target_train, target_test, 8)




###############################################################################
#                                                                             #
#                          Regressor Data Setup                               #
#                                                                             #
###############################################################################

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# tuning_parameter_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

###############################################################################
#                                                                             #
#                     K-Nearest Neighbors Regressor                           #
#                                                                             #
###############################################################################
    
# from sklearn.neighbors import KNeighborsRegressor
# for neighbor_count in range(1,20):
#     knn_model = KNeighborsRegressor(n_neighbors=neighbor_count)
#     knn_model.fit(x_train, y_train)
#     print(neighbor_count,"Neighbors Training Accuracy Score:", str(knn_model.score(x_train,y_train)*100) + "%")
#     print(neighbor_count,"Neighbors Test Accuracy Score:", str(knn_model.score(x_test,y_test)*100) + "%\n")




###############################################################################
#                                                                             #
#                          Linear Regression Regressor                        #
#                                                                             #
###############################################################################

# from sklearn.linear_model import LinearRegression

# print("Linear Regression")
# # l_reg_model = make_pipeline(StandardScaler(), LinearRegression())
# l_reg_model = LinearRegression()
# l_reg_model.fit(x_train, y_train)
# FitAndGetAccuracy(l_reg_model, x_train, x_test, y_train, y_test, 6)



###############################################################################
#                                                                             #
#                          Ridge Regression Regressor                         #
#                                                                             #
###############################################################################

# from sklearn.linear_model import Ridge

# for alpha_val in tuning_parameter_vals:
#     print("Ridge for Alpha=",alpha_val)
#     # ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha_val))
#     ridge_model = Ridge(alpha=alpha_val)
#     FitAndGetAccuracy(ridge_model, x_train, x_test, y_train, y_test, 6)



###############################################################################
#                                                                             #
#                          LASSO Regression Regressor                         #
#                                                                             #
###############################################################################

# from sklearn.linear_model import Lasso

# for alpha_val in tuning_parameter_vals:
#     print("Lasso for Alpha=",alpha_val)
#     # lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=alpha_val))
#     lasso_model = Lasso(alpha=alpha_val)
#     FitAndGetAccuracy(lasso_model, x_train, x_test, y_train, y_test, 6)




###############################################################################
#                                                                             #
#                           Decision Tree Regressor                           #
#                                                                             #
###############################################################################

# from sklearn.tree import DecisionTreeRegressor





###############################################################################
#                                                                             #
#                           Random Forest Regressor                           #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                       Gradient-Boosted Regression Tree                      #
#                                                                             #
###############################################################################




