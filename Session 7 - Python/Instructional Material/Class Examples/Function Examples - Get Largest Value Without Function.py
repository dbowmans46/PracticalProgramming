# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:03:05 2019

MIT License

Copyright (c) 2019 Doug Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

listVals = [10,3,6,8,2,6]
maxValue = None

for val in listVals:
    
    if maxValue == None:
        maxValue = val
    elif val > maxValue:
        maxValue=val
        
print(maxValue)
    
    
listVals = [2,34,789,4456,2,555,-12]

listVals = [-12,-4,-2,-7,-17]

listVals = [3,-1,45,12,9]

listVals = [-3,57,90,-10000,12,0.0001]

listVals = [0.0001,0.0003,0.0008283,0.00000000483]

