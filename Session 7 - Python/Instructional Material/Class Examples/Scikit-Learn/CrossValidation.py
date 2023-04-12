#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris, load_boston

print("Cross Validation on Regressor")
print("------------------------------\n")
data = load_iris()

cv_model = LogisticRegression(max_iter=500000)
cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy")
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())

# The default number of folds is 5.  Can change this with the cv parameter
print("\n\n\n")
print("10 folds")
cvs = cross_val_score(cv_model, data["data"], data["target"], scoring="accuracy", cv=10)
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())

# On regressors, we need to use a different scoring mechanism
print("\n\n\n")
print("Cross Validation on Regressor")
print("------------------------------\n")
data = load_boston()

l_mod = LinearRegression()
cvs = cross_val_score(l_mod, data["data"], data["target"], scoring="r2")
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())

# The default number of folds is 5.  Can change this with the cv parameter
print("\n\n\n")
print("10 folds")
cvs = cross_val_score(l_mod, data["data"], data["target"], scoring="r2", cv=10)
print("Cross Validation Scores: ", cvs)
print("Cross Validation Mean: ", cvs.mean())
print("Cross Validation Standard Deviation: ", cvs.std())