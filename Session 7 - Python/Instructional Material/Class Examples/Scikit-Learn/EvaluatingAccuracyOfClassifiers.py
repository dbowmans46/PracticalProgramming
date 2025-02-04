#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:02:47 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBreastCancerData

# Import metric functions
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Import curves/graphs
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
# The below no longer works as of Scikit-Learn Version 1.2
#from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


# Choose a binary classification data set for simplicity.  1 means the tumor
# is cancerous, 0 means it is not.
data_train, data_test, target_train, target_test = GetTrainTestSplitBreastCancerData()
dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)
dec_tree_model.fit(data_train, target_train)
dec_target_predictions = dec_tree_model.predict(data_test)
dec_target_probabilities_train = dec_tree_model.predict_proba(data_train)
dec_target_probabilities_test = dec_tree_model.predict_proba(data_test)
cancerous_probability_train = [probability[1] for probability in dec_target_probabilities_train]
cancerous_probability_test = [probability[1] for probability in dec_target_probabilities_test]



precision_score_vals = precision_score(target_test, dec_target_predictions)
recall_score_vals = recall_score(target_test, dec_target_predictions)

# Confusion Matrix
# Rows show the actual target values
# Columns show the predictions our model made from test data
confused_matrix = confusion_matrix(target_test, dec_target_predictions)



# Precision vs recall graph.  This method doesn't plot as many points, as it
# interpolates some values.
precisions, recalls, pr_thresholds = precision_recall_curve(target_test, cancerous_probability_test)
plt.scatter(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Graph for Decision Tree Classifier Operating on Breast Cancer Data")
plt.show()

from sklearn.metrics import PrecisionRecallDisplay
prd = PrecisionRecallDisplay(precisions, recalls)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Graph Plotting Directly from PrecisionRecallDisplay Constructor")
prd.plot()
plt.show()

# We can also plot directly from predictions and from estimators
prd = PrecisionRecallDisplay.from_predictions(target_test, dec_target_predictions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Graph Plotting from from_predictions() Method")
prd.plot()
plt.show()

prd = PrecisionRecallDisplay.from_estimator(dec_tree_model, data_test, target_test)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Graph Plotting from from_estimator() Method")
prd.plot()
plt.show()

# Note: The below no longer works as of Scikit-Learn Version 1.2
# There is also a built-in function to auto-generate a graph
# plot_precision_recall_curve(dec_tree_model, data_test, target_test, name = "Dec Tree")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("PR Graph Autogenerated from plot_precision_recall_curve()")
# plt.show()

# ROC Curve
FPR, TPR, roc_thresholds = roc_curve(target_test, cancerous_probability_test)
plt.plot(FPR, TPR)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree Classifier Operating on Breast Cancer Data")
plt.show()

# We can create an ROC display, just like we did with the precision recall
# data
from sklearn.metrics import RocCurveDisplay
# prd = RocCurveDisplay(precisions, recalls)
prd = RocCurveDisplay(fpr=FPR, tpr=TPR)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph Plotting Directly from RocCurveDisplay Constructor")
prd.plot()
plt.show()

# We can also plot directly from predictions and from estimators
prd = RocCurveDisplay.from_predictions(target_test, dec_target_predictions)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph Plotting from from_predictions() Method")
prd.plot()
plt.show()

prd = RocCurveDisplay.from_estimator(dec_tree_model, data_test, target_test)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph Plotting from from_estimator() Method")
prd.plot()
plt.show()

# Note: this no longer works as of Scikit-Learn Version 1.2
# Like the PR curve, we can use the built-in plotter
# plot_roc_curve(dec_tree_model, data_test, target_test, name = "ROC Dec Tree")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve Autogenerated from plot_roc_curve()")
# plt.show()

# AUC of ROC
roc_score = roc_auc_score(target_test, cancerous_probability_test)