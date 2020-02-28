# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:59:31 2020

@author: Doug
"""

import os

writeFilePath = r'C:\Users\Doug\Desktop\test_file.csv'
readFilePath = "C:\\Users\\Doug\\Repositories\\PracticalProgramming\\Session 7 - Python\\Instructional Material\\Class Examples\\File Access Examples\\Part Stock.csv"

dataList = []
# The basic, not preferred method to open a file
# This is not necessarily bad, but it is easier to make mistakes (i.e. forget to close file)
fileHandle = open(writeFilePath,mode='w')
fileHandle.write('This is a bad,way to access files\n')
print("File is closed, and usable by others? ",fileHandle.closed)

## Close the file. Only now will the data get written to the file.
#fileHandle.close()
#print("File is closed, and usable by others now? ",fileHandle.closed)




## Delete file to prepare for next tutorials
#os.remove(writeFilePath)




## Read all the data from the writeFilePath file into a list
#with open(readFilePath, mode='r') as readFileHandle:
#
#    # Iterate through each line of the readFileHandle file
#    for line in readFileHandle.readlines():
#        
#        # Add each line of the file to a list
#        dataList.append(line)
#        
#print(dataList)    
        
        
        
        
## Write the data stored from the read file to a new location
## This file does not exist, but will automatically get created after the next line is ran
## the 'w' token will cause the file to get cleared, if it already exists
#with open(writeFilePath,mode='w') as writeFileHandle:   
#    
#    # Write all elements of the dataList as a line in the new file    
#    writeFileHandle.writelines(dataList)
#    
#    
#    
#    
## Add additional data to the file
## 'a' token allows us to just append data to the file
#with open(writeFilePath,mode='a') as writeFileHandle:  
#    writeFileHandle.write('TestPart,XXXXX\n')
     
    
    
    
## Check what will cause the file to stay open
#with open(writeFilePath,'w') as writeFileHandle:
#    
#    # Inputs cause file to remain locked
#    input("Input here")
#    
#    # Exceptions will still allow file to close
#    raise Exception("Stop code here")
#    
#    print("Line after excepion")
#    
#    # Write all elements of the dataList as a line in the new file    
#    writeFileHandle.writelines(dataList)