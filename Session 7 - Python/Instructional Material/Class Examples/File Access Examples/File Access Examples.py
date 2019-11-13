# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:14:01 2019

MIT License

Copyright (c) 2019 Doug Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


'''
@brief  Print the contents of a file, line by line
@param string filepath The path to the file to get the contents of
'''
def PrintFileContents(filepath):
    
    with open(filepath) as file:
        for line in file.readlines():
            print(line)
    
    return None

filePath = './Project Labor Hours.csv'
PrintFileContents(filePath)










#'''
#@brief Get the project and labor hours froma csv
#@details The project number will be in the first column, the labor hours in the
#         second.
#@param string filepath The path to the file to get the contents of
#'''
#def GetLaborHours(filepath):
#
#    # Dictionary to hold the projects and the hours
#    laborHoursInProjects = {}
#    
#    # Setup the file handle, letting Python know what file to read
#    with open(filepath) as file:
#        
#        # Go through each line in the file
#        for line in file.readlines():
#            
#            # Separate all the values in to specific list elements
#            lineArray = line.split(',')
#            
#            # .strip() removes white space around the value
#            projectNum = lineArray[0].strip()
#            laborHours = lineArray[1].strip()
#            
#            # Test output to make sure we are doing things correctly
#            print(lineArray)
#            print('Project: ' + str(projectNum))
#            print('Labor Hours: ' + str(laborHours))
#            
#            laborHoursInProjects[projectNum] = laborHours
#    
#    # Remove the header row
#    laborHoursInProjects.pop('Project')
#    
#    print(laborHoursInProjects)
#    
#    return None
#
#filePath = './Project Labor Hours.csv'
#GetLaborHours(filePath)













## Example of an overloaded function, to allow the user to specify if the file
## contains headers or not
#
#'''
#@brief Get the project and labor hours froma csv.
#@details The project number will be in the first column, the labor hours in the
#         second.
#@param string filepath The path to the file to get the contents of
#@param bool header Tells if the first row in the file contains headers
#'''
#def GetLaborHours(filepath,header):
#
#    # Dictionary to hold the projects and the hours
#    laborHoursInProjects = {}
#    
#    # Setup the file handle, letting Python know what file to read
#    with open(filepath) as file:
#        
#        # Go through each line in the file
#        for line in file.readlines():
#            
#            # Separate all the values in to specific list elements
#            lineArray = line.split(',')
#            
#            # .strip() removes white space around the value
#            projectNum = lineArray[0].strip()
#            laborHours = lineArray[1].strip()
#            
#            laborHoursInProjects[projectNum] = laborHours
#    
#    # Remove the header row
#    if (header):
#        laborHoursInProjects.pop('Project')
#    
#    print(laborHoursInProjects)
#    
#    return None
#
#filePath = './Project Labor Hours.csv'
#GetLaborHours(filePath,False)