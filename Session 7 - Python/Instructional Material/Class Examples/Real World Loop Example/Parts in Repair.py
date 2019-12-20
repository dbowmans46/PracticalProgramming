# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 07:57:15 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import datetime

# part number is the key, need-by date is the value
partNeedByDates = {
        55293:'20191124',
        101223:'20200312',
        122136:'20180512',
        60500:'20191113',
        121456:'20201010'        
        }

# This is the number of calendar days needed to repair the part, on average, starting from today
partAvgRepairTimeDays = {
        55293:'34',
        101223:'128',
        122136:'212',
        60500:'54',
        121456:'78'
        }

# Task: Assuming people work 7 days a week, determine which parts will get 
# repaired on time and which parts will fall behind.
# Bonus: How many days behind are the parts?

# Hints: 
# 
# datetime library can be used for date and datetime manipulation  
#
# datetime.timedelta can be used to add date objects, such as days, months, or 
# years.  
# 
# To convert a string to a datetime object: 
#     dateTimeVal = datetime.datetime.strptime('19891028','%Y%m%d')
#
# To convert a datetime to a date:
#   dateTimeVal.date()
#
# more info: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior



# TODO: Loop through each part

# TODO: Get when the part is estimated to be completed

# TODO: Check the estimated completion date with the need-by date  

# TODO: Check the date with the lead times to see if the parts will be done on time

# Bonus: How many days behind will the parts be on the estimated completion date?
