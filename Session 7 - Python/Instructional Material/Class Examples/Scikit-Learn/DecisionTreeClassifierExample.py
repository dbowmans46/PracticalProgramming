#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sklearn
from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitIrisData

data = sklearn.load_iris()
data_train, data_test, target_train, target_test = GetTrainTestSplitIrisData()

# This example will use the decision tree classifier
from sklearn.tree import DecisionTreeClassifier

dec_tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)  # Set the classifier type
dec_tree_model.fit(data_train, target_train)            # Train the model with data
dec_target_predictions = dec_tree_model.predict(data_test)

print("Decision Tree Accuracy")
print("----------------------------")
MLHelper.FitAndGetAccuracy(dec_tree_model, data_train, data_test, \
                            target_train, target_test, 8)
print("\n\n")

# Viewing the decision tree
# Create a document that represents the tree
from sklearn.tree import export_graphviz
tree_file_name = "./decision_tree_classifier/decision_tree.dot"
class_name_vals = data.target_names
export_graphviz(dec_tree_model,          # The machine learning model to export \
                out_file=tree_file_name, # The output filepath for the graph \
                class_names=data.target_names,    # Target classes \
                feature_names=data.feature_names, # Feature data names \
                impurity=True, # Show the gini score or not \
                filled=True,   # Fill each node with color in the output image \
                rounded=True)  # Round the corners of the output graph image

# If this does not work due to pathing issues, you can always run dot.exe from
# the Graphviz installation, and generate the graph manually.  See the file
# 'Convert dot.ps1' for an example PowerShell script

# Load the document and view
import graphviz
with open(tree_file_name) as fileHandle:
    dot_graph = fileHandle.read()

graph = graphviz.Source(dot_graph)
s = graphviz.Source(graph.source, filename="test.png", format="png")
s.view()


# How does accuracy change with the number of decisions?
for max_depth_val in range(1,3):
    dtm = DecisionTreeClassifier(max_depth=max_depth_val, random_state=0)  # Set the classifier type
    dtm.fit(data_train, target_train)            # Train the model with data
    dtm_target_predictions = dtm.predict(data_test)
    dtm_score_train = round(dtm.score(data_train,target_train)*100,8)
    dtm_score = round(dtm.score(data_test,target_test)*100,8)
    print("Decision tree training accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score_train) + "%")
    print("Decision tree test     accuracy max_depth=" + str(max_depth_val) + " score:", str(dtm_score) + "%\n")

print(dtm.feature_importances_)
# Why does the accuracy not improve after so many depth levels?  Hint: How many
# attributes do we have?