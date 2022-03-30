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
from sklearn import model_selection

# from sklearn.datasets import load_iris
# iris_dataset = load_iris()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(iris_dataset['data'], 
#                                               iris_dataset['target'], 
#                                               random_state=0)
    
from sklearn.datasets import fetch_california_housing
ca_housing_data = fetch_california_housing()
data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(ca_housing_data['data'], 
                                          ca_housing_data['target'], 
                                          random_state=0)
    
# from sklearn.datasets import load_boston
# boston_data = load_boston()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(boston_data['data'], 
#                                           boston_data['target'], 
#                                           random_state=0)
    
# from sklearn.datasets import load_diabetes
# diabetes_data = load_diabetes()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(diabetes_data['data'], 
#                                           diabetes_data['target'], 
#                                           random_state=0)
   
# from sklearn.datasets import load_breast_cancer
# breast_cancer_data = load_breast_cancer()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(breast_cancer_data['data'], 
#                                           breast_cancer_data['target'], 
#                                           random_state=0)

# from sklearn.datasets import load_wine
# wine_data = load_wine()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(wine_data['data'], 
#                                           wine_data['target'], 
#                                           random_state=0)


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
# MLHelper.FitAndGetAccuracy(dec_tree_model, data_train, data_test, \
#                            target_train, target_test, 8)
# print("\n\n")

# # Viewing the decision tree
# # Create a document that represents the tree
# from sklearn.tree import export_graphviz
# tree_file_name = "./decision_tree_classifier/decision_tree.dot"
# class_name_vals = iris_dataset.target_names
# export_graphviz(dec_tree_model,          # The machine learning model to export \
#                 out_file=tree_file_name, # The output filepath for the graph \
#                 class_names=iris_dataset.target_names,    # Target classes \
#                 feature_names=iris_dataset.feature_names, # Feature data names \
#                 impurity=True, # Show the gini score or not \
#                 filled=True,   # Fill each node with color in the output image \
#                 rounded=True)  # Round the corners of the output graph image 
    
# # If this does not work due to pathin issues, you can always run dot.exe from
# # the Graphviz installation, and generate the graph manually.  See the file
# # 'Convert dot.ps1' for an example PowerShell script

# # Load the document and view
# import graphviz
# with open(tree_file_name) as fileHandle:
#     dot_graph = fileHandle.read()
    
# graph = graphviz.Source(dot_graph)
# s = graphviz.Source(graph.source, filename="test.png", format="png")
# s.view()
    



# # How does accuracy change with the number of decisions?
# for max_depth_val in range(1,3):
#     dtm = DecisionTreeClassifier(max_depth=max_depth_val, random_state=0)  # Set the classifier type
#     dtm.fit(data_train, target_train)            # Train the model with data
#     dtm_target_predictions = dtm.predict(data_test)
#     dtm_score_train = round(dtm.score(data_train,target_train)*100,8)
#     dtm_score = round(dtm.score(data_test,target_test)*100,8)
#     print("Decision tree training accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score_train) + "%")
#     print("Decision tree test     accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score) + "%\n")

# print(dtm.feature_importances_)
# print(breast_cancer_data.feature_names)
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
# lr_model = LogisticRegression(max_iter=1000000)
# lr_model.fit(data_train, target_train)
# target_predictions = lr_model.predict(data_test)
# print("Logistic Regression Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(lr_model, data_train, data_test, \
#                             target_train, target_test, 8)


# c_vals = [0.001]
# for x in range(10):
#     c_vals.append(10*c_vals[-1])
    
# # What happens when we play around with c
# for c_val in c_vals:
#     print("Logistic Regresion with c =",c_val)
#     lr_model = LogisticRegression(C=c_val, max_iter=100000)  # Note that C is capitalized
#     lr_model.fit(data_train, target_train)
#     print("Accuracy Score:", str(lr_model.score(data_test,target_test)*100) + "%")
#     print("\n")
    
    

###############################################################################
#                                                                             #
#                      Linear Support Vector Machine                          #
#                                                                             #
###############################################################################

# from sklearn.svm import LinearSVC

# for c_val in [0.001, 1, 10000]:
#     svc_model = LinearSVC(max_iter=1e7, C=c_val)
#     svc_model.fit(data_train, target_train)
#     target_predictions = svc_model.predict(data_test)
#     print("Linear Support Vector Machine Accuracy")
#     print("----------------------------")
#     MLHelper.FitAndGetAccuracy(svc_model, data_train, data_test,  \
#                                 target_train, target_test, 8)



###############################################################################
#                                                                             #
#                    Non-linear Support Vector Machine                        #
#                                                                             #
###############################################################################



# # Option 1 - Use PolynomialFeatures and LinearSVC to add polynomial features

# from sklearn.svm import LinearSVC
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# # Parameters:
# #    degree - The order to increase each feature, recursively
# #    include_bias - Toggle to include bias column, which contains the intercepts
# poly_features = PolynomialFeatures(degree=3, include_bias=False)
# poly_features_data = poly_features.fit_transform(data_train)
# poly_svm = LinearSVC(loss="hinge", max_iter=1000000)
# poly_svm.fit(poly_features_data, target_train)

# # Can shorten the chain by using Pipelines.  Also, below is a brief peek at
# # scaling data.  Scaling data becomes necessary when the features span vastly
# # different orders of magnitude.  SVM's are particularly sensitive to 
# # scaling
# from sklearn.pipeline import Pipeline
# poly_svm = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
#     ("scalar", StandardScaler()),
#     ("svm_clf", LinearSVC(C=5, loss="hinge", max_iter=1000000))
#     ])
# poly_svm.fit(data_train, target_train)
# print("Linear Support Vector Machine Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(poly_svm, data_train, data_test,  \
#                             target_train, target_test, 8)


# # Option 2 - Use SVC with arguments to use the kernel trick
# from sklearn.svm import SVC

# # coef0 adjusts high-degree polynomial coefficients and low-degree polynomial 
# # coefficients
# nonlinear_svc_model = SVC(kernel="poly", degree=3, coef0=1, max_iter=100000)
# nonlinear_svc_model.fit(data_train, target_train)
# print("Non-linear Polynomial SVM Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(nonlinear_svc_model, data_train, data_test,  \
#                             target_train, target_test, 8)


# # Use SVC with a different kernel
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# # gamma is another regularization parameter.  Larger gamma tightens the kernel
# # boundary around its respective class, while larger gamma values broaden the
# # boundaries.
# rbf_kernel_svc_model = Pipeline([
#     ("scalar", StandardScaler()),
#     ("svm_clf", SVC(kernel="rbf", gamma=5, C=1))
#     ])
# rbf_kernel_svc_model.fit(data_train, target_train)
# print("Non-linear RBF Kernel SVM Accuracy")
# print("----------------------------")
# MLHelper.FitAndGetAccuracy(rbf_kernel_svc_model, data_train, data_test,  \
#                             target_train, target_test, 8)






###############################################################################
#                                                                             #
#                          Regressor Data Setup                               #
#                                                                             #
###############################################################################

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

tuning_parameter_vals = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]



###############################################################################
#                                                                             #
#                     K-Nearest Neighbors Regressor                           #
#                                                                             #
###############################################################################
    
# from sklearn.neighbors import KNeighborsRegressor
# for neighbor_count in range(1,20):
#     knn_model = KNeighborsRegressor(n_neighbors=neighbor_count)
#     knn_model.fit(data_train, target_train)
#     # knn_model = Pipeline([
#     #     ("scalar", StandardScaler()),
#     #     ("knnf", KNeighborsRegressor(n_neighbors=neighbor_count))
#     #     ])
#     # knn_model.fit(data_train, target_train)
#     print(neighbor_count,"Neighbors Training Accuracy Score:", str(knn_model.score(data_train,target_train)*100) + "%")
#     print(neighbor_count,"Neighbors Test Accuracy Score:", str(knn_model.score(data_test,target_test)*100) + "%\n")



###############################################################################
#                                                                             #
#                          Linear Regression Regressor                        #
#                                                                             #
###############################################################################

# from sklearn.linear_model import LinearRegression

# # print("Linear Regression")
# # l_reg_model = make_pipeline(StandardScaler(), LinearRegression())
# # # l_reg_model = LinearRegression()
# # l_reg_model.fit(data_train, target_train)
# # MLHelper.FitAndGetAccuracy(l_reg_model, data_train, data_test, \
# #                             target_train, target_test, 8)
    
# # Time difference in scaling with a pipeline vs normal scaling
# import time
# times = []
# for index in range(500):
#     start_time_standard = time.perf_counter()
#     l_reg_model = LinearRegression()
#     scaled_features = StandardScaler()
#     scaled_features_data = scaled_features.fit_transform(data_train)
#     l_reg_model.fit(scaled_features_data, target_train)
#     # MLHelper.FitAndGetAccuracy(l_reg_model, data_train, data_test, \
#     #                             target_train, target_test, 8)
#     end_time_standard = time.perf_counter()
#     time_standard = end_time_standard - start_time_standard
    
#     start_time_pipeline = time.perf_counter()
#     l_reg_model = make_pipeline(StandardScaler(), LinearRegression())
#     l_reg_model.fit(data_train, target_train)
#     # MLHelper.FitAndGetAccuracy(l_reg_model, data_train, data_test, \
#     #                             target_train, target_test, 8)
#     end_time_pipeline = time.perf_counter()
#     time_pipeline = end_time_pipeline - start_time_pipeline

#     times.append(time_pipeline - time_standard)

# print(sum(times)/len(times))

###############################################################################
#                                                                             #
#                          Ridge Regression Regressor                         #
#                                                                             #
###############################################################################

# from sklearn.linear_model import Ridge

# for alpha_val in tuning_parameter_vals:
#     print("Ridge for Alpha=", alpha_val)
#     ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha_val))
#     # ridge_model = Ridge(alpha=alpha_val)
#     MLHelper.FitAndGetAccuracy(ridge_model, data_train, data_test, \
#                                 target_train, target_test, 8)


###############################################################################
#                                                                             #
#                          LASSO Regression Regressor                         #
#                                                                             #
###############################################################################

# from sklearn.linear_model import Lasso
# import numpy as np

# for alpha_val in tuning_parameter_vals:
#     print("Lasso for Alpha=",alpha_val)
#     #lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=alpha_val))
#     lasso_model = Lasso(alpha=alpha_val)
#     MLHelper.FitAndGetAccuracy(lasso_model, data_train, data_test, \
#                                 target_train, target_test, 8)
        
#     pred_test_lasso = lasso_model.predict(data_test)
#     #print("MSE: ", np.sqrt(mean_squared_error(target_test,pred_test_lasso)))
#     #print("Weights: ", lasso_model.coef_)


###############################################################################
#                                                                             #
#                           Decision Tree Regressor                           #
#                                                                             #
###############################################################################

from sklearn.tree import DecisionTreeRegressor

dec_tree_reg = DecisionTreeRegressor(max_depth=4)
dec_tree_reg.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
                            target_train, target_test, 8)

for depth_val in range(1, 9):
    print("Decision Tree Regressor Depth=",depth_val)
    dec_tree_reg = DecisionTreeRegressor(max_depth=depth_val)
    dec_tree_reg.fit(data_train, target_train)
    MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
                                target_train, target_test, 8)


# Viewing the decision tree
# Create a document that represents the tree
from sklearn.tree import export_graphviz
tree_file_name = "./decision_tree_regressor/decision_tree.dot"
class_name_vals = ca_housing_data.target_names
export_graphviz(dec_tree_reg,            # The machine learning model to export \
                out_file=tree_file_name, # The output filepath for the graph \
                class_names=ca_housing_data.target_names,    # Target classes \
                feature_names=ca_housing_data.feature_names, # Feature data names \
                impurity=True, # Show the gini score or not \
                filled=True,   # Fill each node with color in the output image \
                rounded=True)  # Round the corners of the output graph image 
    
# # If this does not work due to pathin issues, you can always run dot.exe from
# # the Graphviz installation, and generate the graph manually.  See the file
# # 'Convert dot.ps1' for an example PowerShell script

# # Load the document and view
# import graphviz
# with open(tree_file_name) as fileHandle:
#     dot_graph = fileHandle.read()
    
# graph = graphviz.Source(dot_graph)
# s = graphviz.Source(graph.source, filename="test.png", format="png")
# s.view()




###############################################################################
#                                                                             #
#                              Train Test Split                               #
#                                                                             #
###############################################################################

# from sklearn.datasets import load_wine
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression

# wine_data = load_wine()
# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(wine_data['data'], 
#                                           wine_data['target'], 
#                                           random_state=0)

# tts_model = LogisticRegression(max_iter=500000)
# tts_model.fit(data_train, target_train)
# print("TTS Score: ", tts_model.score(data_test, target_test))



###############################################################################
#                                                                             #
#                              Cross Validation                               #
#                                                                             #
###############################################################################

# from sklearn.datasets import load_wine
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score

# wine_data = load_wine()
# cv_model = LogisticRegression(max_iter=500000)
# cvs = cross_val_score(cv_model, wine_data["data"], wine_data["target"], scoring="accuracy")
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())


###############################################################################
#                                                                             #
#                                Scaling Data                                 #
#                                                                             #
###############################################################################






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




###############################################################################
#                                                                             #
#                      Evaluating Predictive Capabilities                     #
#                                                                             #
###############################################################################











