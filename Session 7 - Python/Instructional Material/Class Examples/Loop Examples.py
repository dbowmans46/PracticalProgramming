# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:49:22 2019

MIT License

Copyright (c) 2019 Doug Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

'''
@brief Example function of a while loop.  Get the values of tan(x), 
       possibly for plotting values.
'''
def WhileLoopExample():
    
    import math
    
    x=0
    
    # Iterate until we find a value of x that is the same value as the output 
    # of the tangent of x.  Basically, where the input value equals the output 
    # value for the tangent function
    while(x <= math.tan(x)):
        print('x: ' + str(x))
        print('tan(x): ' + str(math.tan(x)))
        print('\n')
        
        x=x+0.1
        
    #print(type(x))
    # Notice the values inside the loop are still updated, even after the 
    # condition fails.
    print('Outside of Loop:')
    print('x: ' + str(x))
    print('tan(x): ' + str(math.tan(x)))
    
    return None

'''
@brief Example function of using a 
       for loop using a list
'''
def ForLoopExample1():
    
    # This is the collection we are iterating through.  Any collection
    # can be iterated through, such as tuples or lists.
    iterationValues = [0,1,2,3,4,5]
    
    # x is called the loop variable, and takes the place of each element
    # in the collection as the loop traverses each element
    for x in iterationValues:
        
        print(x)
    
    return None


'''
@brief Example function of using a 
       for loop using a range 
'''
def ForLoopExample2():
    
    # range() Creates a list of numbers starting at 0 and incrementing
    # by 1 until it reaches the input value.  The values remain less than the 
    # input value.
    iterationValues2 = range(5,9
    
    # TODO: create an index variable, then use that to iterate through list item
    # iterationValues = [10,11,12,13,14,15]
        
    
    for test in iterationValues2:
        
        print(test)
        
    return None


#WhileLoopExample()
#ForLoopExample1()
ForLoopExample2()