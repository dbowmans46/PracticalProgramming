# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:35:17 2019

MIT License

Copyright (c) 2019 Doug Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

'''
@brief Prints the value passed to the function
@param string valueToPrint
'''
def PrintValue(valueToPrint):
    
    # Prints the argument passed to the function
    print(valueToPrint)
    
    # Explicitly tell the function to return a null value.
    # This is done to prevent sytax errors if there is no code block within the
    # function
    return None

# PrintValue('This is the value to print')
# outputVal = PrintValue('This is the value to print')

'''
@brief Raises the base to a given exponent power
@param number base The base to exponentiate
@param number exponent The power to raise the base to
'''
def Power(base,exponent):
    
    # The return statement directs the statement following 'return' back to code
    # that calls the function, and the function can be used as if it was the 
    # result of the return statement.  For instance, a variable can be used to 
    # store the return value, the value can be printed, etc.  
    return base**exponent

# Power(2,3)
# outputVal = Power(3,3)


'''
@brief Gets all the folders and files on the current user'd desktop
'''
def GetDesktopContainers():
	
	# Operating System library
	import os
	
	# containers is a local variable, and is in the local scope.
	# Once the function returns, containers will be marked for deletion
	containers = []
	
	userName = os.getlogin()
	desktopDirPath = "C:\\users\\" + userName + "\\Desktop\\"
	containers = os.listdir(desktopDirPath)

	return containers
