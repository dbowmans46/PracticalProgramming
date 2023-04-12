#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:27:06 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

csv_filepath = "../../In-Class Exercises/Data/housing.csv"
file_lines = []
with open(csv_filepath, 'r') as fileHandle:
    file_lines = fileHandle.readlines()

# We can then split up the lines into individual elements of lists
csv_data_lines = []
for line in file_lines:
    # We can clean this up a bit by prematurely removing the newline characters
    # before splitting the lines
    csv_data_lines.append(line.replace('\n','').split(','))

print(csv_data_lines)


# Each element is a string.  If we need to, we can preemptively go through and
# convert each to an int, as needed, or just convert on the fly, as needed.