# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:53:52 2020

@author: Doug
"""


import os


dirPath = "C:\\Users\\Doug\\Repositories\\PracticalProgramming\\Session 7 - Python\\Instructional Material\\Class Examples\\File Access Examples\\"
thisIsAList = ["a","B"]

for container in os.scandir(dirPath):    
    
     if (os.path.isfile(container)):
          print(container," is a file")

     elif (os.path.isdir(container)):
          print(container," is a folder")

         

          

# Working with files

textLines = []

with open(dirPath + "Part Stock.csv") as fileHandle:    

     for fileLine in fileHandle.readlines():
          cleanedLine = fileLine.strip()
          
          textLines.append(cleanedLine)
          
          print(cleanedLine)
          
          userInput = input("Stop")

print(textLines)
 



for listItem in textLines:
     print(listItem)
     
     
     
     
     