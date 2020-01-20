# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:59:14 2019

Copyright 2019 Douglas Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Print statements are to separate output in terminal from runfile output that shows
# from running script in Spyder
print("\n")
print("\n")

# Datetimes are defined in the datetime library
import datetime

##########################################
# Converting from a string to a datetime #
##########################################
dateTimeInStringFormat = '20191028'  # In YYYYMMDD format
dateTimeInDatetimeFormat = datetime.datetime.strptime(dateTimeInStringFormat,'%Y%m%d')

#
#print("Conversion to a DateTime")
#print("-----------------------")
#print("String format:    " + dateTimeInStringFormat + ", datatype: " + str(type(dateTimeInStringFormat)))
#print("")
#print("Datetime format: ", dateTimeInDatetimeFormat, ", datatype: ",type(dateTimeInDatetimeFormat))
#print("\n")
#
#
#


######################################################
## Arithmetic with datetimes: use datetime.timedelta #
######################################################
#
## Definition: class datetime.timedelta(days=0, seconds=0, microseconds=0, 
##                                       milliseconds=0, minutes=0, hours=0, weeks=0)
#
## The definition shows that only days, seconds, microseconds, milliseconds, 
## minutes, hours, and/or weeks can be specified
#daysToAdd       = 64
#daysToSubtract  = 44
#monthsToAdd     = 3 # Notice the difference in days when using the average days/month
#yearsToAdd      = 20
#
#WEEKS_PER_YEAR      = 52.1775
#AVG_WEEKS_PER_MONTH = WEEKS_PER_YEAR/12
#AVG_DAYS_PER_MONTH  = 30.44
#AVG_DAYS_PER_YEAR   = 365.25
#
#dateTimeDaysAdded      = dateTimeInDatetimeFormat + datetime.timedelta(days=daysToAdd)
#dateTimeDaysSubtracted = dateTimeInDatetimeFormat - datetime.timedelta(days=daysToSubtract)
## Notice the difference in days when using the average days/month
#dateTimeMonthsAdded    = dateTimeInDatetimeFormat + datetime.timedelta(days=monthsToAdd * AVG_DAYS_PER_MONTH)
#dateTimeYearsAdded     = dateTimeInDatetimeFormat + datetime.timedelta(days=yearsToAdd * AVG_DAYS_PER_YEAR)
#
#print("Arithmetic with DateTimes")
#print("-------------------------")
#print("(", str(dateTimeInDatetimeFormat),  ") +",  daysToAdd,   "  days:", dateTimeDaysAdded)
#print("(", str(dateTimeInDatetimeFormat),  ") -",  daysToSubtract,   "  days:", dateTimeDaysSubtracted)
#print("(", str(dateTimeInDatetimeFormat),  ") +",  monthsToAdd, " months:", dateTimeMonthsAdded)
#print("(", str(dateTimeInDatetimeFormat),  ") +",  yearsToAdd,  " years:", dateTimeYearsAdded)
#print("\n")
#
#
#
#

###########################################################
## Date Constructs Larger than a Week: Use dateutil Library #
#############################################################
#
## If we want to specify date constructs longer than weeks, we can use the library dateutil
## dateutil is not part of the standard library, but can be found here:
##       https://pypi.org/project/python-dateutil/
## Notice the difference when not using averages of days/month or days/year
#
#import dateutil
#
#dateUtilDaysAdded      = dateTimeInDatetimeFormat + dateutil.relativedelta.relativedelta(days=daysToAdd)
#dateUtilDaysSubtracted = dateTimeInDatetimeFormat - dateutil.relativedelta.relativedelta(days=daysToSubtract)
#dateUtilMonthsAdded    = dateTimeInDatetimeFormat + dateutil.relativedelta.relativedelta(months=monthsToAdd)
#dateUtilYearsAdded     = dateTimeInDatetimeFormat + dateutil.relativedelta.relativedelta(years=yearsToAdd)
#
#print("Using dateutil")
#print("--------------")
#print("(", str(dateTimeInDatetimeFormat),  ") +",  daysToAdd,   "  days:", dateUtilDaysAdded)
#print("(", str(dateTimeInDatetimeFormat),  ") -",  daysToSubtract,   "  days:", dateUtilDaysSubtracted)
#print("(", str(dateTimeInDatetimeFormat),  ") +",  monthsToAdd, " months:", dateUtilMonthsAdded)
#print("(", str(dateTimeInDatetimeFormat),  ") +",  yearsToAdd,  " years:", dateUtilYearsAdded)
#print("\n")
#
## dateutil outputs datetimes, so the two libraries are compatible with each other
#print("dateutil produces datetime objects:")
#print("dateUtilDaysAdded type:", type(dateUtilDaysAdded))
#print("dateTimeDaysAdded type:", type(dateTimeDaysAdded))
#print("\n")
#



##########################################################################
## Converting a datetime back to a string - datetime.datetime.strftime() #
##########################################################################
#
#dateTimeStringFromDateTime          = datetime.datetime.strftime(dateTimeDaysAdded,"%m-%d-%Y")
#dateTimeStringFromDateTimeSAPFormat = datetime.datetime.strftime(dateTimeDaysAdded,"%Y%m%d")
#
#print("Converting Datetime to String")
#print("-----------------------------")
#print("(",dateTimeDaysAdded,") converted to string (month-day-year):",dateTimeStringFromDateTime)
#print("(",dateTimeDaysAdded,") converted to SAP string (YYYYMMDD):",dateTimeStringFromDateTimeSAPFormat)
#
