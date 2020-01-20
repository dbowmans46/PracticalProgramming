# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:53:52 2020

@author: Doug
"""


import os


dirPath = "C:\\Users\\Doug\\Repositories\\PracticalProgramming\\Session 7 - Python\\Instructional Material\\Class Examples\\File Access Examples\\"

for x in os.scandir(dirPath):    

     if (os.path.isfile(x)):
          print(x," is a file")

     elif (os.path.isdir(x)):
          print(x," is a folder")

         

          

# Working with files

textLines = []

with open(dirPath + "Part Stock.csv") as fileHandle:    

     for line in fileHandle.readlines():
          cleanedLine = line.strip()
          
          textLines.append(cleanedLine)
          
          print(cleanedLine)

print(textLines)
 

for line in textLines:
     print(line)