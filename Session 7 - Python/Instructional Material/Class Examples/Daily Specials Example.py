# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:30:45 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
This program determins the day of the week, and displays the current
menu special.
"""

# We enlist the help of a library to determine the current date, 
# and also to manipulate dates.  This library is code
# someone else wrote for us to more easily program in Python
import datetime

# This list contains each day of the week
dayOfWeek = ('Monday',
             'Tuesday',
             'Wednesday',
             'Thursday',
             'Friday',
             'Saturday',
             'Sunday')

# This dictionary contains the special for each day of the week
specials = {
        'Monday' : 'Wayne\'s Smoked Brisket',
        'Tuesday' : 'Terra Cotta Biryani',
        'Wednesday' : 'Beijing Duck',
        'Thursday': 'MacTavish\'s Haggis',
        'Friday' : 'Reykjavik Fish and Chips',
        'Saturday' : 'Lowry\'s Prime Rib',
        'Sunday' : 'Delhi Chole'
        }

# Start and end times of happy hour
happyHourTimes = {'Start' : 12,
                  'End' : 13}

# The additional special during happy hour
happyHourSpecials = {
        'Monday' : 'Southern Pecan Pie',
        'Tuesday' : 'Milk Cake',
        'Wednesday' : 'Dong Po Rou',
        'Thursday': 'Cranachan',
        'Friday' : 'Mondlukaka, 1/2 off all cocktails',
        'Saturday' : '1/2 off all Strait Spirits',
        'Sunday' : 'Dahi Chaat'
        }




# Get the current date, and output the value
currentDate = datetime.date.today()
# print('currentDate: ' + str(currentDate))

# Get the current hour
currentDateTime = datetime.datetime.now()
currentHour = currentDateTime.hour

# Get the number of the weekday of the current date, starting at 0 for Monday
# and ending at 6 for Sunday
currentWeekday = datetime.date.weekday(currentDate)
# print('currentWeekday: ' + str(currentWeekday))

# Get the name of the weekday from the number of the weekday
currentWeekdayName = dayOfWeek[currentWeekday]
# print('currentWeekdayName: ' + str(currentWeekdayName))

# Finally, output the daily special(s)
currentSpecial = specials[currentWeekdayName]
# print('currentSpecial: ' + str(currentSpecial))


print('Special for ' + str(currentDate))
print(currentSpecial)

# If we are currently in happy hour, then get and print the happy hour special
currentHappyHourSpecial = happyHourSpecials[currentWeekdayName]
# print('currentHappyHourSpecial: ' + str(currentHappyHourSpecial))

if ( (currentHour >= happyHourTimes['Start']) and (currentHour < happyHourTimes['End'])):
    print(currentHappyHourSpecial)
    