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
# Bonus: Get those who haven't worked a full week.
    
# Create variable to hold times

HOURS_PER_WEEK = 40

# key = employee name
# value = total time worked per week
employeeTotalTimes = {}
    
# TODO: Loop through each key-value pair of the employeeTimes dictionary

for employeeAndTime in employeeTimesThisWeek:
    employee = employeeAndTime[0]
    timeWorked = employeeAndTime[1]    
    
    if (employee in employeeTotalTimes.keys()):
        employeeTotalTimes[employee] += timeWorked
    else:
        employeeTotalTimes[employee] = timeWorked

    
overTime = []
normalTime = []
underTime = []    
for employee in employeeTotalTimes:
    
    timeWorked = employeeTotalTimes[employee]
    
    if employeeTotalTimes[employee] > HOURS_PER_WEEK:
        overTime.append(employee +" " + str(timeWorked))
    elif employeeTotalTimes[employee] == HOURS_PER_WEEK:
        normalTime.append(employee + " " + str(timeWorked))
    else:
        underTime.append(employee + " " + str(timeWorked))
      

print("overTime: " + str(overTime))
print("normalTime: " , normalTime)
print("underTime: " , underTime)
    
    
# TODO: Add each time to the appropriate person
    
# TODO: Manually add the times of one employee to test script functionality
