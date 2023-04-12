#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from MLHelper import MLHelper
from LoadScikitLearnDataSets import GetTrainTestSplitBostonHousingData
from sklearn.datasets import load_boston

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

data = load_boston()
data_train, data_test, target_train, target_test = GetTrainTestSplitBostonHousingData()

tuning_parameter_vals = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]

dec_tree_reg = DecisionTreeRegressor(max_depth=5, \
                                        #max_leaf_nodes=10, \
                                        #max_features=7, \
                                        #min_samples_split=4, \
                                        #min_samples_leaf=1, \
                                        #min_weight_fraction_leaf=1 \
)
dec_tree_reg.fit(data_train, target_train)
MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
                            target_train, target_test, 6)

for depth_val in range(1, 30):
    print("Decision Tree Regressor Depth=",depth_val)
    dec_tree_reg = DecisionTreeRegressor(max_depth=depth_val)
    dec_tree_reg.fit(data_train, target_train)
    MLHelper.FitAndGetAccuracy(dec_tree_reg, data_train, data_test, \
                                target_train, target_test, 8)


# Viewing the decision tree
# Create a document that represents the tree
from sklearn.tree import export_graphviz
tree_file_name = "./decision_tree_regressor/decision_tree.dot"
# class_name_vals = data.target_names
class_name_vals = "Median House Val"
export_graphviz(dec_tree_reg,            # The machine learning model to export \
                out_file=tree_file_name, # The output filepath for the graph \
                class_names=class_name_vals,    # Target classes \
                feature_names=data.feature_names, # Feature data names \
                impurity=True, # Show the gini score/mse or not \
                filled=True,   # Fill each node with color in the output image \
                rounded=True)  # Round the corners of the output graph image

# If this does not work due to pathing issues, you can always run dot.exe from
# the Graphviz installation, and generate the graph manually.  See the file
# 'Convert dot.ps1' for an example PowerShell script.  The graphviz\bin directory
# must be in you environment PATH variable for Python to autoload the graph
# file.

# Load the document and view
import graphviz
with open(tree_file_name) as fileHandle:
    dot_graph = fileHandle.read()


graph = graphviz.Source(dot_graph)
s = graphviz.Source(graph.source, filename="test.png", format="png")
s.view()