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

# Let's set up our first model.  We will use a 75% training to 25% testing data
# ratio.  First, let's separate our training and testing data.  We use the
# train_test_split function to get x and y values (data points and accurate 
# outcomes) for our training data, and our testing data.  This function
# will shuffle the data, using a pseudorandom number generator that takes
# the random_state argument as the seed.
from sklearn import model_selection


def GetTrainTestSplitIrisData():
    """
    Classification data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import load_iris
    data = load_iris()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                                  data['target'], 
                                                  random_state=0)
        
    return data_train, data_test, target_train, target_test


def GetTrainTestSplitCAHousingData():
    """
    Regressor data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'], 
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test


def GetTrainTestSplitBostonHousingData():
    """
    Regressor data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import load_boston
    data = load_boston()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'], 
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test    


def GetTrainTestSplitDiabetesData():
    """
    Regressor data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'], 
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test   


def GetTrainTestSplitBreastCancerData():
    """
    Classification data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'], 
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test


def GetTrainTestSplitWineData():
    """
    Classification data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import load_wine
    data = load_wine()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'],
                                              test_size=0.25,
                                              train_size=0.75,
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test


def GetTrainTestSplitCovertypeData():
    """
    Classification data set

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data['data'], 
                                              data['target'],
                                              test_size=0.25,
                                              train_size=0.75,
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test


def GetHousingData():    
    """
    Regression data set from https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv

    Returns
    -------
    data_train : TYPE
        DESCRIPTION.
    data_test : TYPE
        DESCRIPTION.
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    """
    
    import pandas as pd
    
    data_file_path = "../../In-Class Exercises/Data/housing.csv"
    data = pd.read_csv(data_file_path)
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(data.drop('median_house_value',axis=1), 
                                              data['median_house_value'],
                                              test_size=0.25,
                                              train_size=0.75,
                                              random_state=0)
        
    return data_train, data_test, target_train, target_test



# print("\nIntro Model - K-Nearest Neighbors Classifier\n")
# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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
# class_name_vals = data.target_names
# export_graphviz(dec_tree_model,          # The machine learning model to export \
#                 out_file=tree_file_name, # The output filepath for the graph \
#                 class_names=data.target_names,    # Target classes \
#                 feature_names=data.feature_names, # Feature data names \
#                 impurity=True, # Show the gini score or not \
#                 filled=True,   # Fill each node with color in the output image \
#                 rounded=True)  # Round the corners of the output graph image 
    
# # If this does not work due to pathing issues, you can always run dot.exe from
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

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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
#     print("Logistic Regression with c =",c_val)
#     lr_model = LogisticRegression(C=c_val, max_iter=100000)  # Note that C is capitalized
#     lr_model.fit(data_train, target_train)
#     print("Accuracy Score:", str(lr_model.score(data_test,target_test)*100) + "%")
#     print("\n")
    
    

###############################################################################
#                                                                             #
#                      Linear Support Vector Machine                          #
#                                                                             #
###############################################################################

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

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

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# data_train, data_test, target_train, target_test = GetTrainTestSplitBostonHousingData()

# tuning_parameter_vals = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]



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

# from sklearn.tree import DecisionTreeRegressor

# dec_tree_reg = DecisionTreeRegressor(max_depth=5, \
#                                         #max_leaf_nodes=10, \
#                                         #max_features=7, \
#                                         #min_samples_split=4, \
#                                         #min_samples_leaf=1, \
#                                         #min_weight_fraction_leaf=1 \
# )
# dec_tree_reg.fit(data_train, target_train)
# MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
#                             target_train, target_test, 6)

# for depth_val in range(1, 30):
#     print("Decision Tree Regressor Depth=",depth_val)
#     dec_tree_reg = DecisionTreeRegressor(max_depth=depth_val)
#     dec_tree_reg.fit(data_train, target_train)
#     MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
#                                 target_train, target_test, 8)


# # Viewing the decision tree
# # Create a document that represents the tree
# from sklearn.tree import export_graphviz
# tree_file_name = "./decision_tree_regressor/decision_tree.dot"
# # class_name_vals = data.target_names
# class_name_vals = "Median House Val"
# export_graphviz(dec_tree_reg,            # The machine learning model to export \
#                 out_file=tree_file_name, # The output filepath for the graph \
#                 class_names=class_name_vals,    # Target classes \                
#                 feature_names=data.feature_names, # Feature data names \
#                 impurity=True, # Show the gini score/mse or not \
#                 filled=True,   # Fill each node with color in the output image \
#                 rounded=True)  # Round the corners of the output graph image 
    
# If this does not work due to pathing issues, you can always run dot.exe from
# the Graphviz installation, and generate the graph manually.  See the file
# 'Convert dot.ps1' for an example PowerShell script.  The graphviz\bin directory
# must be in you environment PATH variable for Python to autoload the graph
# file.

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


# # test_size + train_size <= 1.00
# sklearn.model_selection.train_test_split(data['data'], 
#                                           data['target'], 
#                                           test_size=0.3,
#                                           train_size=0.7,
#                                           random_state=0)

# from sklearn.linear_model import LogisticRegression
# tts_model = LogisticRegression(max_iter=500000)
# MLHelper.FitAndGetAccuracy(tts_model, data_train, data_test, \
#                             target_train, target_test, 4)



###############################################################################
#                                                                             #
#                              Cross Validation                               #
#                                                                             #
###############################################################################

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_iris, load_boston

# print("Cross Validation on Regressor")
# print("------------------------------\n")
# data = load_iris()

# cv_model = LogisticRegression(max_iter=500000)
# cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy")
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())

# # The default number of folds is 5.  Can change this with the cv parameter
# print("\n\n\n")
# print("10 folds")
# cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy", cv=10)
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())

# # On regressors, we need to use a different scoring mechanism
# print("\n\n\n")
# print("Cross Validation on Regressor")
# print("------------------------------\n")
# data = load_boston()

# l_mod = LinearRegression()
# cvs = cross_val_score(l_mod, data["data"], data["target"], scoring="r2")
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())

# # The default number of folds is 5.  Can change this with the cv parameter
# print("\n\n\n")
# print("10 folds")
# cvs = cross_val_score(l_mod, data["data"], data["target"], scoring="r2", cv=10)
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())



###############################################################################
#                                                                             #
#                                Scaling Data                                 #
#                                                                             #
###############################################################################

# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

# C_val = 1000
# max_iterations = 100000

# print("Accuracy without scaling:\n","--------------------\n")
# log_reg_model = LogisticRegression(C=C_val, max_iter=max_iterations)
# MLHelper.FitAndGetAccuracy(log_reg_model, data_train, data_test, \
#                             target_train, target_test, 8)
    

# print("\n\n\n")
# print("Accuracy with scaling:\n","--------------------\n")

# scaler = StandardScaler()
# # scaler = RobustScaler()
# # scaler = MinMaxScaler()
# scaled_train_data = scaler.fit_transform(data_train, target_train)
# scaled_test_data = scaler.fit_transform(data_test, target_test)
# log_reg_model = LogisticRegression(C=C_val, max_iter=max_iterations)
# MLHelper.FitAndGetAccuracy(log_reg_model, scaled_train_data, scaled_test_data, \
#                             target_train, target_test, 8)
    
# print("\n\n\n")
# print("Accuracy with scaling and cross validation:\n","--------------------\n")

# from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score

# data = load_iris()
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data['data'], data['target'])
# cvs = cross_val_score(log_reg_model, scaled_data, data["target"], scoring="accuracy")
    
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())


# # Scaling using a Pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# print("\n\n\n")
# print("Accuracy with Scaling and Using Pipelines:")

# pipeline_w_scaler = Pipeline([
#     ("scalar", StandardScaler()),
#     ("log_reg", LogisticRegression(C=C_val, max_iter=max_iterations))
#     ])
# MLHelper.FitAndGetAccuracy(pipeline_w_scaler, data_train, data_test, \
#                             target_train, target_test, 8)
# cvs = cross_val_score(pipeline_w_scaler, data["data"], data["target"], scoring="accuracy")
# print("Cross Validation Scores: ", cvs)
# print("Cross Validation Mean: ", cvs.mean())
# print("Cross Validation Standard Deviation: ", cvs.std())


    
###############################################################################
#                                                                             #
#                       Converting Categories to Numbers                      #
#                                                                             #
###############################################################################

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

# # Load data that has category-based values
# data_file_path = "../../In-Class Exercises/Data/housing.csv"
# data_set = pd.read_csv(data_file_path)

# le = LabelEncoder()
# encoded_ocean_data = le.fit_transform(data_set['ocean_proximity'])

# # Remove the string-based data and put in the encoded data
# data_encoded = data_set.drop('ocean_proximity', axis=1)
# data_encoded['ocean_proximity'] = encoded_ocean_data
# print(data_encoded.head(10))





# # One-hot encoding the data
# # This will build off of the numerically-categorized data we produced above
# ohe = OneHotEncoder()

# # The fit_transform method expects a 2D matrix, so we need to convert the 
# # encoded data from a 1D array to a 2D array.
# encoded_ocean_data_reshaped = encoded_ocean_data.reshape(-1,1)
# ohe_ocean_data = ohe.fit_transform(encoded_ocean_data_reshaped)

# # By default, one-hot encoding returns a SciPy sparse matrix to save memory.
# # To add this to our dataframe, we can convert it to a normal array.
# ohe_ocean_data_arr = ohe_ocean_data.toarray()

# # # If we transpose the matrix, the data will be in a format easier to make
# # # new columns with for our Pandas dataframe
# ohe_ocean_data_arr = ohe_ocean_data_arr.transpose()

# # Now we can make a column for each category name.  To check what each category
# # number corresponds to, we can use the "classes_" property.  We can also
# # use each label directly, as we will to later.
# print("Label categories:", le.classes_)

# # As before, we can remove the old string-category column
# data_ohe_encoded = data_set.drop('ocean_proximity', axis=1)

# # Now we can add the new columns for each category, and add the appropriate
# # data.

# for category_index in range(len(le.classes_)):
#     category_name = le.classes_[category_index]
#     print("Category to add to the dataframe: ", category_name)
#     data_ohe_encoded[category_name] = ohe_ocean_data_arr[category_index]
    
# print(data_ohe_encoded.head(1))



# # For sparse matrices, you will need to adapt the code to convert the data
# # for it to work with a dataframe.  There are functions that can import
# # a sparse matrix into a dataframe directly.



# # We can use the LabelBinarizer to accomplish the label encoding, and the 
# # one-hot encoding at the same time.  This will produce separate arrays
# # that we will still need to manipulate into a form that a Pandas dataframe
# # will like.
# lb = LabelBinarizer()
# ocean_cat_lb = lb.fit_transform(data_set['ocean_proximity'])

# # We can check if this output is the same as our previous output after 
# # applying the label encoding and one-hot encoding sequentially
# print("LabelBinarizer Check:", ocean_cat_lb == ohe_ocean_data)


###############################################################################
#                                                                             #
#                           Estimating Missing Values                         #
#                                                                             #
###############################################################################

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder

# # Load data that has category-based values
# data_file_path = "../../In-Class Exercises/Data/housing.csv"
# data_set = pd.read_csv(data_file_path)

# # Imputers cannot estimate text, so first we convert string data to a number.
# # Reminder: scikit-learn algorithms may erroneously assume numbers that are
# # closer together are more correlated.  To avoid this, use OneHotEncoder
# # or LabelBinarizer, as seen in a previous example.
# le = LabelEncoder()
# encoded_ocean_data = le.fit_transform(data_set['ocean_proximity'])
# data_encoded = data_set.drop('ocean_proximity', axis=1)
# data_encoded['ocean_proximity'] = encoded_ocean_data

# # Now that we have numerical data, we can impute missing values.
# si = SimpleImputer(strategy="median")
# imputed_data = si.fit(data_encoded)

# # We can take a look at the median values by looking at the statistics
# print(imputed_data.statistics_)

# # Now we apply the median value to the NaN values.
# filled_in_data = imputed_data.transform(data_encoded)

# # Did it work?  Here are some checks
# # 1. make sure the data sizes are the same
# print("Size check: ", data_set.shape == data_encoded.shape)

# # 2. Spot check a known NaN value.  Can also compare values of entire row to ensure
# # the correct row was compared.
# print("Dataset NaN at row 538:", data_set['total_bedrooms'].iloc[538])
# print("Imputed data at row 538:",  filled_in_data[538][4])

# # The filled in data is in an array that we can make a DataFrame with if we want
# filled_in_df = pd.DataFrame(data=filled_in_data, columns=data_set.columns)

# # We can now split the data into training data and test data, and train a
# # machine learning model as above.
# #
# # Notice that the ocean_proximity column median values are all 1, representing
# # INLAND.  If this is most likely not the case, you could either pick a different
# # strategy, or remove the records that have NaN values.


###############################################################################
#                                                                             #
#                         Checking Feature Correlation                        #
#                                                                             #
###############################################################################

# # Pandas dataframe correlation matrix, same as Pearson's R correlation factor
# # Only useful for linear correlations
# import pandas as pd
# # data_file_path = "../../In-Class Exercises/Data/housing.csv"
# data_file_path = "../../In-Class Exercises/Data/housing NaN Strings.csv"
# original_data_set = pd.read_csv(data_file_path)
# print("Correlation matrix: ")
# print(original_data_set.corr())

# # We can also check the correlation of a single feature with all others
# print("Correlations of just median_house_value:")
# print(original_data_set.corr()["median_house_value"])
# print("\n\n")





# # This is Pearson's R correlation factor, same as the DataFrame.corr() method.
# # Only useful for linear correlations
# import scipy
# pearson_r, p_val = scipy.stats.pearsonr(original_data_set["median_house_value"], original_data_set["latitude"])
# print("pearson_r:", pearson_r)
# print("p_val:", p_val,"\n\n")

# # This is Spearman's R correlation factor, and can be used for non-linear 
# # correlations
# spearman_r, p_val = scipy.stats.spearmanr(original_data_set["median_house_value"], original_data_set["latitude"])
# print("spearman_r:", spearman_r)
# print("p_val:", p_val,"\n\n\n")





# # In addition to correlation factors, feature importances can tell us how 
# # much a feature affects targets.  For this example, we will also need to 
# # convert string data to numerical, imputemissing values, and manually split 
# # off the data and targets from a DataFrame

# # First, convert string categories to numbers
# from sklearn.preprocessing import LabelBinarizer
# lb = LabelBinarizer()
# ocean_cat_lb = lb.fit_transform(original_data_set['ocean_proximity'])

# # Transpose for easy addition to dataframe
# ocean_cat_lb = ocean_cat_lb.transpose()

# # Drop string categorical data
# data_encoded = original_data_set.drop('ocean_proximity', axis=1)

# # Add numerical categorical data
# for category_index in range(len(lb.classes_)):
#     category_name = lb.classes_[category_index]
#     data_encoded[category_name] = ocean_cat_lb[category_index]

# # Let's replace the invalid values.  We will use the Imputer estimator as we saw above
# # to estimate these values.
# from sklearn.impute import SimpleImputer
# si = SimpleImputer(strategy="median")
# imputed_data = si.fit(data_encoded)
# filled_in_data = imputed_data.transform(data_encoded)

# # Let's create a dataframe to house our new data, and give each column an 
# # appropriate name.
# # Since we added new columns, need to incorporate those into the list of columns
# # for the new dataframe
# new_col_names = original_data_set.columns.tolist()
# new_col_names.remove('ocean_proximity')
# new_col_names.extend(lb.classes_)
# filled_in_df = pd.DataFrame(data=filled_in_data, columns=new_col_names)

# # Set up the targets and the data
# targets = filled_in_df["median_house_value"]
# data =  filled_in_df.drop(labels="median_house_value", axis=1)

# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(data, 
#                                           targets, 
#                                           random_state=0)

# # Train a model that can utilize feature_importances_ (any of the decision tree
# # regressors will work)
# from sklearn.tree import DecisionTreeRegressor

# dtr = DecisionTreeRegressor()
# dtr.fit(data_train, target_train)

# # And finally, we can check the feature importance
# col_names = filled_in_df.columns.tolist()
# col_names.remove("median_house_value")
# print("Feature Importances:")
# for index in range(len(dtr.feature_importances_)):
#     # Create an index holder.  Will be sliced to align text after index.
#     index_str = str(index) + ")   "
    
#     # Used to align feature importance values
#     spacer_string = "                          "
#     len_to_strip = len('housing_median_age') - len(col_names[index])
#     print(index_str[:5], col_names[index], spacer_string[:len_to_strip], dtr.feature_importances_[index])







###############################################################################
#                                                                             #
#           Dimensionality Reduction - Principal Component Analysis           #
#                                                                             #
###############################################################################

# from sklearn.decomposition import PCA

# # For this example, we will use a data set with fewer dimensions
# from sklearn.datasets import load_iris
# iris_dataset = load_iris()
# iris_data = iris_dataset['data']

# # Note that for this example, the data is all within the same order of magnitude,
# # and thus we do not need to scale the data. If this is not the case, the data 
# # should first be scaled before passing it  to the PCA transformer.  For
# # sake of completeness, the data will be scaled.
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# scaled_data = sc.fit_transform(iris_data)

# # For this basic example, we will just reduce the number of dimensions by 1
# principal_component_count = iris_data.shape[1] - 1
# pca = PCA(n_components = principal_component_count)
# pca_transformed_data = pca.fit_transform(scaled_data)
# print(pca_transformed_data)

# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(pca_transformed_data, 
#                                               iris_dataset['target'], 
#                                               random_state=0)

# # For this example, the KNN classifier is used for no particular reason, 
# # and the number of neighbors chosen for no particular reason, as well.
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=3) 
# MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)  

# # We can get the original features by inversing the transform
# pca_inverted_data = pca.inverse_transform(pca_transformed_data)

# # We can also see how much of the data we have lost by checking how much data
# # is preserved in our reduced model.  This information is held in the 
# # explained_variance_ratio_ member.  We can see that we still have about 99.5%
# # of the total variance in the reduced data, so using the reduced feature set
# # is a good tradeoff (remember, keeping at least 95% is ta good starting 
# # point).

# print("Explained Variance Ratios: ", pca.explained_variance_ratio_)
# print("Total variance captured in the reduced model: ", sum(pca.explained_variance_ratio_))



# # We can also let the algorithm determine how many dimensions we should keep
# # based on the variance we want to maintain
# min_variance = 0.95

# # The only difference is using the fraction of fariance as the input to the PCA
# # argument
# pca = PCA(n_components = min_variance)
# pca_transformed_data = pca.fit_transform(scaled_data)
# print(pca_transformed_data)

# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(pca_transformed_data, 
#                                               iris_dataset['target'], 
#                                               random_state=0)

# # For this example, the KNN classifier is used for no particular reason, 
# # and the number of neighbors chosen for no particular reason, as well.
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=3) 
# MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)  

# print("\n\n\nSetting Variance Fraction")
# print("-------------------------------")
# print("Explained Variance Ratios: ", pca.explained_variance_ratio_)
# print("Total variance captured in the reduced model: ", sum(pca.explained_variance_ratio_))







###############################################################################
#                                                                             #
#     Dimensionality Reduction - Incremental Principal Component Analysis     #
#                                                                             #
###############################################################################

from sklearn.decomposition import IncrementalPCA
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()
iris_data = iris_dataset['data']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_data = sc.fit_transform(iris_data)

num_batches = 10
# We cannot use a minimum variance input for incremental PCA, as we could with
# the normal PCA transformer.  Here, we must specify how many components we
# want.
component_count = iris_data.shape[1] - 2
incremental_pca = IncrementalPCA(n_components = component_count)

# We will use NumPy to split the data into equal batches, then feed the
# incremental PCA transformer
for batch_of_data in np.array_split(scaled_data, num_batches):
    incremental_pca.partial_fit(batch_of_data)

# Once we have trained the incremental PCA model, we can use it to transform
# our initial scaled data
data_reduced = incremental_pca.transform(scaled_data)

print("Reduced data: \n", data_reduced)
print("explained variance: ", incremental_pca.explained_variance_ratio_)
print("Variance Maintained: ", sum(incremental_pca.explained_variance_ratio_))
print()

# Now we can use the transformed data as we would have with any other data set.
# First split the data for training and testing, then feed a predictor
data_train, data_test, target_train, target_test = \
    sklearn.model_selection.train_test_split(data_reduced, 
                                              iris_dataset['target'], 
                                              random_state=0)

# For this example, the KNN classifier is used for simplicity.
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3) 
MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)  




###############################################################################
#                                                                             #
#       Dimensionality Reduction - Random Principal Component Analysis        #
#                                                                             #
###############################################################################

# # This example is almost identical to the first with PCA, we are just going 
# # to pass an additional argument to the PCA transformer.  

# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
# iris_dataset = load_iris()
# iris_data = iris_dataset['data']

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# scaled_data = sc.fit_transform(iris_data)

# principal_component_count = iris_data.shape[1] - 2

# # Here is where the only difference occurs, using a different solver with PCA.
# pca = PCA(n_components = principal_component_count, svd_solver='randomized')
# pca_transformed_data = pca.fit_transform(scaled_data)
# print(pca_transformed_data)

# data_train, data_test, target_train, target_test = \
#     sklearn.model_selection.train_test_split(pca_transformed_data, 
#                                               iris_dataset['target'], 
#                                               random_state=0)

# # For this example, the KNN classifier is used for no particular reason, 
# # and the number of neighbors chosen for no particular reason, as well.
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=3) 
# MLHelper.FitAndGetAccuracy(knn_model, data_train, data_test, target_train, target_test)  





###############################################################################
#                                                                             #
#       Dimensionality Reduction - Kernel Principal Component Analysis        #
#                                                                             #
###############################################################################








###############################################################################
#                                                                             #
#                             Dataset Collection                              #
#                                                                             #
###############################################################################



















###############################################################################
#                                                                             #
#                          Manual Feature Engineering                         #
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











