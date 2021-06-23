#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 00:04:40 2018

@author: doug
"""

"""
@brief Test to see the scope of local variables used in recursive functions
"""
class TestClass():
    
    def __init__(self):    
        self.iterationLimit = 5
        self.iterationCount = 0

    def FirstFunction(self):
        testList = [5]
        
        if (self.iterationCount < self.iterationLimit):            
            self.iterationCount += 1
            
            testList.append(self.iterationCount)
            
            print(testList)
            self.SecondFunction()
            
            
        return
    
    
    def SecondFunction(self):
        testList = []
        print(testList)
        self.FirstFunction()
        
        
testClass = TestClass()
testClass.FirstFunction()
    