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
@brief Example function of a while loop
'''
def WhileLoopExample():
    
    import math
    
    x=0
    
    while(x <= math.tan(x)):
        print(x)
        x=x+0.1
    
    return None

'''
@brief Example function of using a 
       for loop using a list
'''
def ForLoopExample1():
    
    iterationValues = [0,1,2,3,4,5]
    
    for x in iterationValues:
        
        print(x)
    
    return None


'''
@brief Example function of using a 
       for loop using a range 
'''
def ForLoopExample2():
    
    iterationValues2 = range(5)
    
    for x in iterationValues2:
        
        print(x)
        
    return None


# WhileLoopExample()
ForLoopExample1()
ForLoopExample2()