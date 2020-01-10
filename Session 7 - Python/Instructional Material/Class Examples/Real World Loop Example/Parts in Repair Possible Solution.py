)
# To convert a string to a datetime object: 
#   dateTimeVal = datetime.datetime.strptime('19891028','%Y%m%d')
#
# To convert a datetime to a date:
#   dateTimeVal.date()



# TODO: Loop through each part

# TODO: Get when the part is estimated to be completed

# TODO: Check the estimated completion date with the need-by date  

# TODO: Check the date with the lead times to see if the parts will be done on time

# Bonus: How many days behind will the parts be on the estimated completion date?


import datetime

behindParts = {}

for part in partNeedByDates:
    
    needByDate = datetime.datetime.strptime(partNeedByDates[part],'%Y%m%d').date()
    partRepairDays = int(partAvgRepairTimeDays[part])
    todaysDate = datetime.date.today()
    
    completionDate = todaysDate + datetime.timedelta(days=partRepairDays)
    
    # Check values while we iterate
    print("Part: " + str(part))
    print("completion date: " + str(completionDate))
    print("Need By Date: " + str(needByDate))
    
    if completionDate > needByDate:
        behindParts[part] = completionDate - needByDate
        
    print()

# Print the days behind
print("\n\n\n")
print("Results")
print("----------------------------")
for part in behindParts:
    daysBehind = behindParts[part]
    print("Part: " + str(part) + ", Days Behind: " + str(daysBehind.days))
