# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:16:02 2021

@author: Doug
"""

from sklearn.metrics import mean_squared_error

class MLHelper():
    
    def FitAndGetAccuracy(model, x_train, x_test, y_train, y_test, dec_points=4):
        """
        Print accuracies for a ML model

        Parameters
        ----------
        model : TYPE
            The scikit-learn model, fitted with data.
        x_train : TYPE
            Training data.
        x_test : TYPE
            Test training data.
        y_train : TYPE
            Training labels.
        y_test : TYPE
            Test labels.
        dec_points : TYPE, optional
            The number of decimal places to display in accuracy measures. The default is 4.

        Returns
        -------
        None.

        """
    
        
        model.fit(x_train, y_train)
        
        train_score = model.score(x_train,y_train)
        test_score  = model.score(x_test, y_test)
        mse         = mean_squared_error(y_test, model.predict(x_test))
        
        formatted_train_score = str(round(train_score,dec_points)*100) + "%"
        formatted_test_score  = str(round(test_score, dec_points)*100) + "%"
        
        print("Train Accuracy Score:", formatted_train_score)
        print("Test Accuracy Score:", formatted_test_score)
        print("MSE Score:", mse)
        print()
        
        return None