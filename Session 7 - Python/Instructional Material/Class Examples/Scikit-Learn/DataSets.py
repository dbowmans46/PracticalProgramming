# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:18:10 2022

@author: Doug
"""

import sklearn
from sklearn.model_selection import train_test_split

class DataSets():
    
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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None
    
    
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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
        
        return None


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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None


    def GetTrainTestSplitDiabetesgData():
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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None


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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None


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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None


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
        DataSets.data_train = data_train
        DataSets.data_test = data_test
        DataSets.target_train = target_train
        DataSets.target_test = target_test
            
        return None