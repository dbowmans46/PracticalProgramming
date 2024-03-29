# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 07:43:03 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# Project is the key, profit is the value
projects = {
        
        90210 : -1029384,
        33950 : 100,
        202449 : 120455,
        339204 : 0,
        'x' : -1400600200,       
        'Krakatoa' : 1800344956.01
        }

# Dictionary will hold the project, and the profit for each project that is in the red
projectDict = {}    

# Loop through each project, checking if the profit is below zero
for project in projects:
    
    # Determine which projects have negative profit
    if projects[project] <= 0:
        
        # Store the projects with negative profit in the dictionary holding
        # projects in the red
        projectDict[x]= projects[x]    
        
    
# Print out the projects that are in the red
print(projectDict)
