# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:13:31 2020

@author: Doug
"""

# TODO: basic variables
#x = 5
#x = "6"
#x = 5.6
#print(type(int(x)))

# TODO: tuple
#x = [1,2,3]
#print(x)
#x.
#print(x)
# TODO: list
# TODO: dictionary
#scores = {"Joe":"50"}
#print(scores["Joe"])

# TODO: Conditions
#condition_value = 60
#x=0
#
#if (condition_value < 5):
#    print(x)
#    x = x - 1
#elif (condition_value > 10):
#    print("Greater than 10")    
#else:    
#    x = x + 1
#    print(x)
#    
#print(x)

# TODO: Loops
#x = [1,2,3,4,5]
#
#for temp_val in x:
#    
#    print(temp_val)
#    print(x)
#    print("\n")
    
#conditional_value = True
#counter = 0
#
#while (conditional_value):
#    print(counter)
#    counter += 1 # counter = counter + 1
#    
#    if (counter > 100):
#        conditional_value = False
    


# TODO: Functions

#def Power(base, power):
#    
#    powerVal = base**power
#    
#    return powerVal
#
#print(Power(2,3))
#
## TODO: Write to a file that doesn't exist
#filePath = "C:/users/doug/desktop/file.txt"
#
#with open(filePath,"a") as fileHandle:
#    
#    fileHandle.write("this is text")
#    
#    
    

## TODO: Read from a file
#tempList = []
#with open(filePath,"r") as fileHandle:
#    
#    for line in fileHandle.readlines():
#        tempList.append(line)
#        
#print(tempList)


#
#filePath = "C:/users/doug/desktop/file.txt"
#with open(filePath,"w") as fileHandle:
#    
#    fileHandle.write("Separate new text")
    
# TODO: Append data to a file


testStr = "this\tis\ta\tstring"

print(testStr.split("\t"))


testStr2 = "this is a test string, as well\n"
print(testStr2.strip())


revenue = "500"
cost = "200\n"
print( int(revenue) - int(cost.split()[0]) )












