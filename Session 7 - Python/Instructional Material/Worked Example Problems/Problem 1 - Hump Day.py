# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:41:07 2020

Copyright 2018 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""






"""
Problem Description:
    
    Determine if the current day of the week is past "hump" day (Wednesday).  
    Assume the first day of the week is Monday, and the last day of the week is 
    Sunday.

Plan of Attack:
    1. Get the current date
    2. Get the number corresponding to the current day of the week
    3. Check if the current day has passed Wednesday
    
Required Libraries:
    Since we are working with dates, the "datetime" library will be useful

"""

import datetime

# First, we hardcode the number corresponding to Wednesday, with Monday starting
# at 0
wed_day_num = 2

# 1. Get the current date
current_date = datetime.datetime.now()

# 2. Get the number corresponding to the current day of the week
current_day_num = datetime.datetime.weekday(current_date)

# 3. Check if the number of the current day of the week is larger than that
# of Wednesdays, and report.
# Optionally, we can report if we have not passed hump day
if (current_day_num > wed_day_num):
    print("Passed hump day!")
else:
    print("Still not over the hump")










