# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:33:34 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# Easily handles Excel data
import pandas

# Used to pause system execution
import time

dataFileLocation = "PartArrivalListener.xlsx"

# Get the data from Excel
partArrivalData = pandas.read_excel(dataFileLocation)

# Get the arrival date
arrivalDate = partArrivalData['DateTime Arrived in Stores'].iloc[2]

# Continually check to see if the DateTime of the part arrival time has been added
# Keep searching until the field is filled out
while (pandas.isnull(arrivalDate)):
    
    # Reload the data to see if the date has been added
    partArrivalData = pandas.read_excel(dataFileLocation)
    arrivalDate = partArrivalData['DateTime Arrived in Stores'].iloc[2]
    
    # Look to see if the DateTime has been added, check every 1 second
    time.sleep(1)

print('Part has arrived: ' + ' ' + str(arrivalDate))