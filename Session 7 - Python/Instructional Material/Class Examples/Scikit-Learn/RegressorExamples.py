# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:26:26 2022

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn

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
# must be in you envinronment PATH variable for Python to autoload the graph
# file.

# # Load the document and view
# import graphviz
# with open(tree_file_name) as fileHandle:
#     dot_graph = fileHandle.read()


# graph = graphviz.Source(dot_graph)
# s = graphviz.Source(graph.source, filename="test.png", format="png")
# s.view()