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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], test_size=0.25, train_size=0.75, random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data["data"], data["target"], test_size=0.25, train_size=0.75, random_state=0
    )

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
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data.drop("median_house_value", axis=1),
        data["median_house_value"],
        test_size=0.25,
        train_size=0.75,
        random_state=0,
    )

    return data_train, data_test, target_train, target_test

def GetTrainTestSplitMakeMoonsData():
    """
    Classifier data set

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
    
    from sklearn.datasets import make_moons
    
    data, targets = make_moons(n_samples=1000, noise=0.25)
    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = sklearn.model_selection.train_test_split(
        data, targets, test_size=0.25, train_size=0.75, random_state=0
    )

    return data_train, data_test, target_train, target_test
