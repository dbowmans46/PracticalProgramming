# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:27:08 2022

@author: Doug
"""


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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

data = load_iris()

cv_model = LogisticRegression(max_iter=500000)
cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy")
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())

# # The default number of folds is 5.  Can change this with the cv parameter
# print("\n\n\n")
# print("10 folds")
# cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy", cv=10)
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

# C_val = 100
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
    
###############################################################################
#                                                                             #
#                       Converting Categories to Numbers                      #
#                                                                             #
###############################################################################




###############################################################################
#                                                                             #
#                           Estimating Missing Values                         #
#                                                                             #
###############################################################################




###############################################################################
#                                                                             #
#                         Checking Feature Correlation                        #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                          Manual Feature Engineering                         #
#                                                                             #
###############################################################################