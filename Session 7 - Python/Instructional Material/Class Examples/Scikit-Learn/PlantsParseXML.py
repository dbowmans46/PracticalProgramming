#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:19:55 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import urllib.request

# The xml package handles reading XML structures in Python
import xml.etree.ElementTree as ET

# XML sources
plants_xml_url = "https://www.w3schools.com/xml/plant_catalog.xml"

xml_text = ""

# First, let's get the XML data from the website
fhand = urllib.request.urlopen(plants_xml_url)
for line in fhand:
    # Information coming from the website is in byte format.  We need to decode
    # it into a text format that can be stored in a string
    xml_text += line.decode().strip()

# Next, let's put it in a format Python can effectively parse
xml_tree = ET.fromstring(xml_text) # Can use ET.parse() if there is a local file to read

plant_item_template = {"common":"",
                       "botanical":"",
                       "zone":"",
                       "light":"",
                       "price":"",
                       "availability":""}

plants = []
for plant_item in xml_tree:
    
    plant = plant_item_template.copy()
    
    for parameter_item in plant_item:
       plant[parameter_item.tag.lower()] = parameter_item.text
        
    plants.append(plant)
    print(plant['common'], "-", plant['zone'])

plants_original = plants.copy()
iteration_len = len(plants)-1

for plant in plants:
    #print("iteration len: ", iteration_len)
    for plant_index in range(iteration_len):
        if plants[plant_index]['zone'] > plants[plant_index+1]['zone']:
            #print(plants[plant_index]['zone'] )
            #print(plants[plant_index+1]['zone'] )
            temp_var = plants[plant_index+1]['zone']
            plants[plant_index+1]['zone'] = plants[plant_index]['zone']
            plants[plant_index]['zone'] = temp_var
            #print(plants[plant_index]['zone'] )
            #print(plants[plant_index+1]['zone'] )
            #print()
            
    iteration_len-=1
        
    
    
