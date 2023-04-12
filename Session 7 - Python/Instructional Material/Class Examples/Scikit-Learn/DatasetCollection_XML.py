#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:31:41 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Library to read HTTP data/handle requests
import urllib.request

# The xml package handles reading XML structures in Python
import xml.etree.ElementTree as ET

# XML sources
breakfast_menu_xml_url = "https://www.w3schools.com/xml/simple.xml"
# cd_collection_xml_url = "https://www.w3schools.com/xml/cd_catalog.xml"
# plants_xml_url = "https://www.w3schools.com/xml/plant_catalog.xml"
# embedded_xml_text_url = "https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms762271(v=vs.85)"

xml_text = ""

# First, let's get the XML data from the website
fhand = urllib.request.urlopen(breakfast_menu_xml_url)
for line in fhand:
    # Information coming from the website is in byte format.  We need to decode
    # it into a text format that can be stored in a string
    xml_text += line.decode().strip()

# Next, let's put it in a format Python can effectively parse
xml_tree = ET.fromstring(xml_text) # Can use ET.parse() if there is a local file to read

# Can search for specific elements within the node
prices = []
for menu_item in xml_tree:
    prices.append(menu_item.findall('price'))
    
for price in prices:
    # Each price is a single-element list
    print(price[0].text)
print("\n\n\n")


# Get all the items from the data.  Could also write a recursive function
# to get all data from a general data structure

# Setup the basic structure of each menu item's data in a template we will
# copy later.
food_menu_item_template = {"name":"",
                            "price":0,
                            "description":"",
                            "calories":0}
food_menu_items = []
# Get each set of data pertaining to the breakfast menu options
for menu_item in xml_tree:
    food_menu_item = food_menu_item_template.copy()
    
    # Get the details for each menu option
    for item_parameters in menu_item:
        food_menu_item[item_parameters.tag] = item_parameters.text
        
    food_menu_items.append(food_menu_item)

# Output the data structure for testing purposes
for menu_item in food_menu_items:
    for key_val in menu_item:
        print(key_val, ":", menu_item[key_val])
        
    print("\n")
print("\n\n\n")

        
        
# XML Attributes and parsing files
breakfast_xml_filepath = "../../In-Class Exercises/Data/breakfast_menu_modified.xml"
xml_tree_from_file = ET.parse(breakfast_xml_filepath)
xml_tree_root = xml_tree_from_file.getroot()

# First, setup a new template for storing the data
food_menu_item_template = {"name":"",
                            "price":0,
                            "description":"",
                            "calories":0,
                            "attribute":{}}

# Get each set of data pertaining to the breakfast menu options
food_menu_items = []
for menu_item in xml_tree_root:
    food_menu_item = food_menu_item_template.copy()
    
    # Get the details for each menu option    
    temp_attribute = {}
    for item_parameters in menu_item:
        food_menu_item[item_parameters.tag] = item_parameters.text
    
        # Create a temporary dictionary for the attributes, only
        # if there is actual attribute data for this menu item
        if len(item_parameters.attrib.keys()) > 0:
            for key in item_parameters.attrib.keys():
                temp_attribute[key] = item_parameters.attrib[key]
        food_menu_item['attribute'] = temp_attribute
        
    food_menu_items.append(food_menu_item)

# Check data parsed
for menu_item in food_menu_items:
    print("Name: ", menu_item['name'])
    print("Price: ", menu_item['price'])
    print("Description: ", menu_item['description'])
    print("Calories: ", menu_item['calories'])
    print("Attributes: ", menu_item['attribute'])
    print("\n")

# XML namespaces
import xml.etree.ElementTree as ET

xml_file_path = "../../In-Class Exercises/Data/namespace_example.xml"

# if there is a special encoding for the file, we may need to specify that when
# reading the file
utf_parser = ET.XMLParser(encoding="utf-8")
xml_tree = ET.parse(xml_file_path, parser=utf_parser)

print("Getting all tags:")
# To use a namespace, use the URL enclosed in curly braces before the tag
# you are searching for.
for tree in xml_tree.findall('{http://arborists.example.com}tree'):
    for tag in tree:
        print(str(tag.tag)+ ":", tag.text)

print("\n")
print("Getting just a single tag:")
# We can get a specific hierarchy using findall multiple times.  Remember, 
# findall() only searches one hierarchy level.
for tree in xml_tree.findall('{http://arborists.example.com}tree'):
    for tag in tree.findall('{http://arborists.example.com}tree_type'):
        print(str(tag.tag)+ ":", tag.text)

# To make it easier for using the namespaces, we can setup a dictionary
xmlns_dict = {"arborist":"http://arborists.example.com",
              "data_struct":"http://data_structs.example.com"
              }

print("\n")
print("Using a namespace dictionary:")
# root = fromstring(xml_text)
for tree in xml_tree.findall('data_struct:tree', xmlns_dict):
    for tag in tree:
        print(str(tag.tag)+ ":", tag.text)

