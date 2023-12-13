#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:35:49 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Library to read HTTP data/handle requests
import urllib.request

# Beautiful soup can parse HTML
from bs4 import BeautifulSoup

ms_xml_url = 'https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms762271(v=vs.85)'

# Request web text
html_text = ""
fhand = urllib.request.urlopen(ms_xml_url)
for line in fhand:
    print(line.strip())
    html_text += line.decode().strip("\n")
    print()
    print(line.decode().strip("\n"))

# Load HTML text from website into BeautifulSoup
soup = BeautifulSoup(html_text, 'html.parser')

# Now, we need to get the XML string.  We will pull the text from the <code> 
# tag, amd use the .string property to convert the HTML encoding of special 
# characters to their ascii representation (i.e. '&gt;' to '>')

# Either of these will work
#xml_navigable_str = soup.find_all('code')[0].string
xml_navigable_str = soup('code')[0].string

xml_string = str(xml_navigable_str)

import xml.etree.ElementTree as ET
xml_tree = ET.fromstring(xml_string)


output_struct = {"book_id":"",
                 "author":"",
                 "title":"",
                 "genre":"",
                 "price":0.0,
                 "publish_date":"",
                 "description":""
                 }

output_file_text = ""

# Add headers
output_file_text = "book_id,author,title,genre,price,publish_date,description\n"

output_line = ""
for book in xml_tree:
    # Clear the string before adding new line of data
    output_line = ""
    book_id = book.attrib["id"]
    output_struct["book_id"] = '"' + book_id + '",'
    for book_item in book:
        output_struct[book_item.tag] = '"' + book_item.text + '",'
    
    output_line = output_struct["book_id"] + output_struct["author"] + \
                    output_struct["title"] + output_struct["genre"] + \
                    output_struct["price"] + output_struct["publish_date"]
                    
    # We don't want the last comma, so take a string slice
    # We also need to replace multiple spaces with a single one
    output_line +=  output_struct["description"][:-1].replace("      "," ").replace("  "," ") + "\n"
    print(output_line)
    output_file_text += output_line
    
# Output text to CSV
filepath = "XML Problem 3 Output.csv"
with open(filepath,'w') as file_handle:
    file_handle.write(output_file_text)
