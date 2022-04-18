# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:25:37 2022

@author: Doug
"""


import sklearn

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
