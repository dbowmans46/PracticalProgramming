#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:16:14 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
import pandas
from sklearn import model_selection

###############################################################################
#                                 Gather Data                                 #
###############################################################################

# We will use the fire alarm inspection data to try and predict
# which ID's have had a recent annual inspection.
filepath = "../../In-Class Exercises/Data/Detroit Fire Alarm Inspection/Fire_Inspections.csv"
data = pandas.read_csv(filepath)
#data = data.set_index('IO_ID')

###############################################################################
#                               Data Preparation                              #
###############################################################################

# There are quite a few records with missing lat and lon values
# I am choosing to keep those in for now, since missing coordinates may be an
# indicator in which ID's do not have up-to-date fire alarm inspections.
# These values do, however, need to be converted to numbers
import numpy as np
data = data.replace(np.nan,0)

# We can see that the addresses are mostly unique, and the ones that are the 
# same don't necessarily mean anything.  Drop this column.
data = data.drop(['Address', 'address_id'], axis=1)

# The propusetypedescription is just a description string of the propusetype
# column, so drop this one
data = data.drop('propusetypedescription', axis=1)

# X and Y are the same as latitutde and longitude, so drop those
data = data.drop(['X', 'Y'], axis=1)

# The occupant name and structure nameinfo is encoded in the IO_ID column, so 
# drop these string columns
data = data.drop('OccupantName', axis=1)
data = data.drop('StructureName', axis=1)

# propusetype has NNN and UUU values.  These need converted into a value.
# We could alternatively drop these rows, if there aren't too many.  Further,
# we may need to label binarize these values to pevent the trainer from making
# erroneous associations with the numbers.
data = data.replace({'propusetype':'UUU'}, value = 2000)
data = data.replace({'propusetype':'NNN'}, value = 5000)

# The last thing we need to modify is the LatestInspDate column.  I will just
# convert this to a single integer-compatible string of YYYYMMDDHHMMSS
def convert_date_time_to_int(date_time_val):
    
    date, time = date_time_val.split(" ")
    date = date.replace("/","")
    time = time.replace(":","").replace("+00","")
    
    return date + time

data['LatestInspDate'] = data['LatestInspDate'].apply(convert_date_time_to_int)



# We need to convert the remaining string data into number data.
# For InspectionType_Full and InspWithinLastYear, these are labels
# and can be encoded.  We will first prepare the InspectionType_Full binary data
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
inspection_type_lb_arr = lb.fit_transform(data['InspectionType_Full'])
# Add the new binary columns with the name of labels
inspection_type_df = pandas.DataFrame(data=inspection_type_lb_arr, columns=lb.classes_)
data_concat_df = pandas.concat([data.reset_index(drop=True), inspection_type_df.reset_index(drop=True)], axis=1)
# Remove the old column, since the data has been label binarized
data_concat_df = data_concat_df.drop('InspectionType_Full', axis=1)

# Now we will label binarize InspWithinLastYear.  Since this is a single column,
# we don't need to concatenate DataFrames together
inspection_last_year_lb_arr = lb.fit_transform(data_concat_df['InspWithinLastYear'])
data_concat_df["InspWithinLastYear"] = inspection_last_year_lb_arr

# Split the data into training data and test data
data_points = data_concat_df.drop('InspWithinLastYear', axis=1)
targets = data_concat_df['InspWithinLastYear']

data_train, data_test, target_train, target_test = \
    model_selection.train_test_split(data_points, targets, random_state=0)




###############################################################################
#                                 Data Scaling                                #
###############################################################################




###############################################################################
#                             Classifier Selection                            #
###############################################################################

# We will try a bunch of classifiers to see which one works best

# This fails with Scikit-Learn v 1.3.0, there is a bug https://github.com/scikit-learn/scikit-learn/issues/26768
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(data_train, target_train)         
target_predictions = knn_model.predict(data_test)

from sklearn.tree import DecisionTreeClassifier
dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)  # Set the classifier type
dec_tree_model.fit(data_train, target_train)            # Train the model with data
dec_target_predictions = dec_tree_model.predict(data_test)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=10000, C=0.005)
lr_model.fit(data_train, target_train)
target_predictions = lr_model.predict(data_test)

from sklearn.svm import LinearSVC
svc_model = LinearSVC(max_iter=10000, C=0.05)
svc_model.fit(data_train, target_train)

from sklearn.svm import SVC
nonlinear_svc_model = SVC(kernel="rbf", gamma=5, C=1, probability=True)
nonlinear_svc_model.fit(data_train, target_train)

from sklearn.ensemble import VotingClassifier
lr_model = LogisticRegression(max_iter=100, n_jobs=-1)
dec_tree_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
# Seting the probability=True for the SVC trainer allows us to utilize soft 
# voting.
svc_model = SVC(probability=True) 

estimators_list = [('lr', lr_model),
                    ('dtc', dec_tree_model),
                    ('knn', knn_model),               
                    ('svc', nonlinear_svc_model)]

voting_model = VotingClassifier(estimators = estimators_list, voting='soft')
voting_model.fit(data_train, target_train)

from sklearn.ensemble import BaggingClassifier
bagger_model = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), 
                                  n_estimators=300, max_samples=75, bootstrap=True, 
                                  n_jobs=-1, oob_score=True)
bagger_model.fit(data_train, target_train)

paster_model = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), 
                                  n_estimators=300, max_samples=75, bootstrap=False, 
                                  n_jobs=-1, oob_score=False)
paster_model.fit(data_train, target_train)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=6000, max_leaf_nodes=15, n_jobs=-1)
rf_model.fit(data_train, target_train)

from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(dec_tree_model, 
                                n_estimators=5, 
                                algorithm="SAMME.R", 
                                learning_rate = 0.5)
ada_model.fit(data_train, target_train)

from sklearn.ensemble import GradientBoostingClassifier
gbt_model = GradientBoostingClassifier(max_depth=2, learning_rate=0.1, subsample=0.27)
gbt_model.fit(data_train, target_train)

###############################################################################
#                                   Metrics                                   #
###############################################################################

import matplotlib.pyplot as plt

# ml_model = knn_model
# ml_model = dec_tree_model
# ml_model = lr_model
# ml_model = svc_model
# ml_model = nonlinear_svc_model
# ml_model = voting_model
# ml_model = bagger_model
# ml_model = paster_model
# ml_model = rf_model
# ml_model = ada_model
# ml_model = gbt_model


# # Setup predictions and probabilities for metric calculations
# model_predictions = ml_model.predict(data_test)
# model_probabilities = ml_model.predict_proba(data_test)
# model_probabilities_pos_class_test_data = [probability[1] for probability in model_probabilities]

# # Confusion matrix 
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# confused_matrix = confusion_matrix(target_test, model_predictions)
# cm_display = ConfusionMatrixDisplay.from_estimator(ml_model, data_test, target_test)
# plt.xlabel("Predictions")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix with Heat Map Scheme")
# plt.show()


# from sklearn.metrics import precision_score, recall_score
# precision_score_vals = precision_score(target_test, model_predictions)
# recall_score_vals = recall_score(target_test, model_predictions)
# print("Precision Score:", precision_score_vals)
# print("Recall Score:", recall_score_vals)

# # Precisoin recall curve
# from sklearn.metrics import PrecisionRecallDisplay
# prd = PrecisionRecallDisplay.from_estimator(ml_model, data_test, target_test)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("PR Graph Plotting from from_estimator() Method")
# plt.show()

# # ROC Curve
# from sklearn.metrics import RocCurveDisplay, roc_auc_score
# prd = RocCurveDisplay.from_estimator(ml_model, data_test, target_test)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Graph Plotting from from_estimator() Method")
# plt.show()


# roc_score = roc_auc_score(target_test, model_probabilities_pos_class_test_data)
# print("ROC Score:", roc_score)

