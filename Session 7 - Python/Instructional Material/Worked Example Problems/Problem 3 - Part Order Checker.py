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
    
    Given the data in "Part Min Qty.csv", determine if a part needs to be ordered 
    to maintain adequate stock and how much needs to be ordered to maintain the 
    safety stock amount.  The "Order Point" is determines when an order needs 
    placed to order more parts. "Safety Stock" is the amount of parts that should 
    be in stores after a part has reached the order point.  For instance, if we 
    have 5 of part A, and the order point is 10, and the safety stock is 15, we 
    need to order 10 more parts so that the stock will be 15.

Plan of Attack:
    
Required Libraries:

"""