# -*- coding utf-8 -*-
"""
Created on Tue Nov 12 07:51:01 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Employee name is the key, time worked is the value
employeeTimesThisWeek = (
        ('Bob' , 8),
        ('Sally', 8),
        ('Reinholdt' , 10),
        ('Jack Ryan' , 4),
        ('Joan' , 12),
        ('Weatherby' , 2),
        ('Scilia' , 10),
        ('Sally', 5),
        ('Jack Ryan' , 12),
        ('Weatherby' , 4),
        ('Joan' , 10),
        ('Joan' , 8),
        ('Sally', 12),
        ('Reinholdt' , 12),
        ('Bob' , 10),
        ('Reinholdt' , 8),
        ('Joan' , 10),
        ('Jack Ryan' , 5),
        ('Sally', 10),
        ('Bob' , 10),
        ('Sally', 6),
        ('Reinholdt' , 8),
        ('Reinholdt' , 9),
        ('Joan' , 10)
        )
    

# Task: Determine which employees have worked overtime this week.
# Bonus: Get the slackers
    
# TODO: Create variable to hold times
    
# TODO: Loop through each key-value pair of the employeeTimes dictionary
    
# TODO: Add each time to the appropriate person
    
# TODO: Manually add the times of one employee to test script functionality

employeeTotalTimes = {}
normalHoursPerWeek = 40

for workerAndTime in employeeTimesThisWeek:
    
    if workerAndTime[0] in employeeTotalTimes.keys():
        employeeTotalTimes[workerAndTime[0]] += workerAndTime[1]
    else:
        employeeTotalTimes[workerAndTime[0]] = workerAndTime[1]
        
        
        
# Sort employees by how much they worked
overTimeWorkers = {}
normalTimeWorkers = {}
underTimeWorkers = {}

# Create an initial record for comparison, setting the time worked higher than 
# any possible time a person can work
lastPlace = {'NoOne':24*7+1}  

for employee in employeeTotalTimes:
    
    timeWorked = employeeTotalTimes[employee]
    
    if timeWorked > normalHoursPerWeek:
        overTimeWorkers[employee] = timeWorked
    elif timeWorked == normalHoursPerWeek:
        normalTimeWorkers [employee] = timeWorked
    else:
        underTimeWorkers[employee] = timeWorked
        
    if timeWorked < lastPlace[list(lastPlace.keys())[0]]:
        lastPlace = {employee:timeWorked}

print("Overtime Workers:")        
print(overTimeWorkers)
print()

print("Normal Work Week Workers:")
print(normalTimeWorkers)
print()

print("Under Achievers:")
print(underTimeWorkers)
print()

print("Least time worked:")
print(lastPlace)
print()