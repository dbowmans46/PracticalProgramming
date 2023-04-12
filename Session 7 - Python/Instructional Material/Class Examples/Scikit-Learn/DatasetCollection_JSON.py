#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:31:44 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Library to read HTTP data/handle requests
import urllib.request

# Using the json library to convert JSON data to Python structures
import json

json_widget_file_path = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Data/Example Widget Data.json"
json_webapp_file_path = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Data/Example Webapp Data.json"

print("Widget Example")
with open(json_widget_file_path, 'r') as jsonFileHandle:
    json_text = jsonFileHandle.read()

# We can then create a JSONDecoder to convert the string into Python lists, 
# dictionaries, and other data structures.
json_decoder = json.JSONDecoder() 
json_structures = json_decoder.decode(json_text)
print(json_structures)
print()

# From here, we can traverse the structure like we would any nested Python
# data structures
print(json_structures['widget']['window'])
print("\n\n\n")


# # A more complicated example can include both dictionaries and lists nested
# # in the structure
# print("Webapp Example")
# with open(json_webapp_file_path, 'r') as jsonFileHandle:
#     json_text = jsonFileHandle.read()

# # We can then create a JSONDecoder to convert the string into Python lists, 
# # dictionaries, and other data structures.
# json_decoder = json.JSONDecoder() 
# json_structures = json_decoder.decode(json_text)

# # See file layout for a look at the entire data structure in a neatly formatted
# # style.
# print(json_structures['web-app']['servlet'][1])
# print("\n\n\n")


# # We can use urllib to pull text from the web, and then parse it like we would
# # any normal string.
# # Note: be careful blindly trying to read .json files.  Reading some files will
# # over utilize resources on a computer, and crash it.
# print("Reading json files from the internet")
# json_fruit_url = "https://support.oneskyapp.com/hc/en-us/article_attachments/202761627/example_1.json"
# #json_quiz_url = "https://support.oneskyapp.com/hc/en-us/article_attachments/202761727/example_2.json"
# #json_housing_fhfa_data_url = "https://www.fhfa.gov/HPI_master.json"
# url_handle = urllib.request.urlopen(json_fruit_url)
# json_web_string = ""
# for line in url_handle:
#     # Information coming from the website is in byte format.  We need to decode
#     # it into a text format that can be stored in a string
#     json_web_string += line.decode().strip()

# # We can take a look at the web string to see what we have
# print("String pulled from internet:")
# print(json_web_string)
# print()

# json_decoder = json.JSONDecoder() 
# json_fruit_structure = json_decoder.decode(json_web_string)
# print("Fruit: ", json_fruit_structure['fruit'])
# print()


# json_quiz_url = "https://support.oneskyapp.com/hc/en-us/article_attachments/202761727/example_2.json"
# url_handle = urllib.request.urlopen(json_quiz_url)
# json_web_string = ""
# for line in url_handle:
#     # Information coming from the website is in byte format.  We need to decode
#     # it into a text format that can be stored in a string
#     json_web_string += line.decode().strip()
    
# # We can take a look at the web string to see what we have
# print("String pulled from internet:")
# print(json_web_string)
# print()

# json_decoder = json.JSONDecoder() 
# json_quiz_structure = json_decoder.decode(json_web_string)
# print("Quiz math questions: ", json_quiz_structure['quiz']['maths'])
# print("\n\n\n")

