# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:41:27 2020

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
    
    Check if the email addresses in the file "Emails to Check for Addition.csv" 
    exist in the file "Employee Email Addresses.csv". If not, add them.  Ensure 
    the email addresses are saved in alphabetical order in the file "Employee 
    Email Addresses.csv".

Plan of Attack:
    
    1. Read email addresses already stored, store in list
    2. Read email addresses to add/check if already exists, store in list
    3. For each email to add, check if it exists.  If not, added it to the list 
       of stored email addresses.
    4. Sort the final list of email addresses
    5. Write the list to "Employee Email Addresses.csv"
    
Required Libraries:
    None
    
"""

# Create a function that will clean the lines of the files
'''
@brief Clean lines from a .csv file
@param string csvFileLine A line from a .csv file
'''
def CleanCSVLine(csvFileLine):
    
    # Remove whitespace at the beginning and end of each line
    cleanedLine = csvFileLine.strip()
    
    return cleanedLine


# Create a function that will read the contents of a .csv file, and store it
# in a list
def ReadSingleColCSVFile(filePath):
    
    dataList = []
    
    # Open the file in read-only mode
    with open(filePath,'r') as fileHandle:
        
        # Clean each line, and add it to the output data list
        for line in fileHandle.readlines():
            cleanedLine = CleanCSVLine(line)
            dataList.append(cleanedLine)
    
    # Return the list to where the function was called
    return dataList
    


# Setup file names 
currentEmailListFilePath = "Employee Email Addresses.csv"
emailsToAddFilePath = "Emails to Check For Addition.csv"
updatedEmailsFilePath = "Updated Email Addresses.csv"

# 1. Read email addresses already stored, store in list
# Use the function created above
currentEmailAddresses = ReadSingleColCSVFile(currentEmailListFilePath)

# Store the header for formatting the output file
headerLabel = currentEmailAddresses[0]

# Drop the header line form the .csv file
currentEmailAddresses.remove(currentEmailAddresses[0])



# 2. Read email addresses to add/check if already exists, store in list
# Use the function created above
emailAddressesToAddList = ReadSingleColCSVFile(emailsToAddFilePath)

# Drop the header line form the .csv file
emailAddressesToAddList.remove(emailAddressesToAddList[0])




# 3. For each email to add, check if it exists.  If not, added it to the list 
#    of stored email addresses.
for emailToAdd in emailAddressesToAddList:
    if(emailToAdd not in currentEmailAddresses):
        currentEmailAddresses.append(emailToAdd)

# 4. Sort the final list of email addresses
currentEmailAddresses.sort()


# 5. Write the list to "Employee Email Addresses.csv"
with open(updatedEmailsFilePath,'w') as fileHandle:
    
    # Write the header line
    fileHandle.write(headerLabel + "\n")
    
    # Write each email address as a new line in the output .csv file, adding the
    # new line character to fulfill the format of the.csv file
    for emailAddress in currentEmailAddresses:
        fileHandle.write(emailAddress + "\n")



