# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:16:02 2021

@author: Doug
"""

from sklearn.metrics import mean_squared_error

class MLHelper():
    
    def FitAndGetAccuracy(model, x_train, x_test, y_train, y_test, dec_places=4):
        """
        Print accuracies for a ML model

        Parameters
        ----------
        model : sci-kit-learn-predictor
            The scikit-learn model, fitted with data.
        x_train : tuple
            Training data.
        x_test : tuple
            Test training data.
        y_train : tuple
            Training labels.
        y_test : tuple
            Test labels.
        dec_places : int, optional
            The number of decimal places to display in accuracy measures. The default is 4.

        Returns
        -------
        None.

        """
    
        
        model.fit(x_train, y_train)
        
        train_score = model.score(x_train,y_train)
        test_score  = model.score(x_test, y_test)
        mse         = mean_squared_error(y_test, model.predict(x_test))
        
        formatted_train_score = str(round(train_score*100, dec_places)) + "%"
        formatted_test_score  = str(round(test_score*100, dec_places)) + "%"
        formatted_mse = str(round(mse, dec_places))
        
        print("Train Accuracy Score:", formatted_train_score)
        print("Test Accuracy Score:", formatted_test_score)
        print("MSE Score:", formatted_mse)
        print()
        
        return None
    
    
    def FormatScore(score_val, dec_places):
        """
        

        Parameters
        ----------
        score_val : TYPE
            DESCRIPTION.
        dec_places : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return str(round(score_val*100, dec_places)) + "%"