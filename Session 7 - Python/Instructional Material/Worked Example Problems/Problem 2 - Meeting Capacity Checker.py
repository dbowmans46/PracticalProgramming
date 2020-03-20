# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:41:26 2020

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
    
    Given the meeting room maximum capacities in "Room Capacities", have the user 
    input a room and the number of employees that will use it, and return whether 
    the room will be below, at, or over capacity.

Plan of Attack:
    1. Read the contents of the file
    2. Filter/clean data as necessary
    3. Store data in a dictionary for comparison
    4. Ask user for room name as input
    5. Ask user for number of attendees using room
    6. Check if the room can accommodate, and report
    
Required Libraries:
    None, only standard library functionality utilized

"""

# Set the path of the file.  Creating a variable for this makes it easier to 
# update all future references to the file path vs hardcoding in each separate
# location
filePath = "./Room Capacities.csv"

# Setup dictionary to hold room name and capacity
roomCapacityDict = {}

# 1. Read the contents of the file

with open(filePath) as fileHandle:
    
    # Store each line in the dictionary, cleaning as necessary
    for line in fileHandle.readlines():
       # 2. Filter/clean data as necessary

       # Don't store the first line of the file, as this is a header
       if (line[0:4] != "Room"):
           # Remove white space from line, including new line character
           lineCleaned = line.strip()
           
           # Separate the line in to two parts, store as list
           lineSplit = lineCleaned.split(",")
           
           roomName = lineSplit[0]
           roomCapacity = lineSplit[1]
           
           # 3. Store data in a dictionary for comparison
           roomCapacityDict[roomName] = roomCapacity

# Optional: Print list of available rooms for the user
print("Available Rooms:")
for roomName in roomCapacityDict.keys():
    print(roomName)

# Add line separator between room names and user input
print("\n")



# 4. Ask user for room name as input
# Optional: Ensure user selects a room that exists
userRoomNameInput = ""
while (userRoomNameInput not in roomCapacityDict.keys()):
    
    # Ask for and store user input
    userRoomNameInput = input("Which room will you be using: ")
    
    # If the room the user is trying to use does not exist, inform the user.
    # When the loop cycles through, the user will be asked to input the room again
    if (userRoomNameInput not in roomCapacityDict.keys()):
        print("Room name not found.  Please select the name of a room within the building. Available names are as follows: ")
        
        # This is a list comprehension, a short-hand form of a for loop
        [print(room) for room in roomCapacityDict.keys()]



# 5. Ask user for number of attendees using room
userCapacityInput = input("How many members will attend: ")

# 6. Check if the room can accommodate, and report
chosenRoomCapacity = roomCapacityDict[userRoomNameInput]

# Optional: Output user's data and room data
print("Chosen room: ",userRoomNameInput)
print("Attendees: ", userCapacityInput)
print("Room Limit: ", chosenRoomCapacity)

# Finally, report if the room can accommodate the employees or not
if (int(chosenRoomCapacity) < int(userCapacityInput)):
    print("Number of attendees exceeds room capacity")
else:
    print("Room can accommodate number of attendees")
       
       
"""
Ending Notes:
    
    This example shows some user input, as well as how to check for simple
    errors that may occur due to the user not inputting correct values.
    
    The code also shows how to filter/clean data from an ideal .csv file.  User
    input and data errors require more focused scrubbing.
"""    
    
       
