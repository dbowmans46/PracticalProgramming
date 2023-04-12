#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:31:34 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Library to read HTTP data/handle requests
import urllib.request

# Beautiful soup can parse HTML
from bs4 import BeautifulSoup

html_url = "http://www.williams-int.com/"
romeo_url = "http://data.pr4e.org/romeo.txt"
pypi_url = "https://pypi.org/"

# Request web text
html_text = ""
fhand = urllib.request.urlopen(html_url)
for line in fhand:
    html_text += line.decode().strip()

# Load HTML text from website into BeautifulSoup
soup = BeautifulSoup(html_text, 'html.parser')

# We can output the text in a human-readable format, such as a programmer 
# would format the text as
print("Formatted Text:")
print(soup.prettify())
print("\n\n\n")

# We can filter on specific elements
print("Get title and HTML tags: ", soup.title)
print("\n")
print("Get just the HTML title tag's name:", soup.title.name)
print("\n")
print("Get just the title text:", soup.title.string)
print("\n")
print("Get the HTML tag that contains the title element:", soup.title.parent.name)
print("\n")
print("Get the first HTML paragraph tag:", soup.p)
print("\n")
print("Get all HTML paragraph tags, as a list:", soup.find_all('p'))
print("\n")
print("Get each HTML paragraph tag:")
for paragraph in soup.find_all('p'):
    print(paragraph.prettify())
print("\n")
print("Get the first HTML reference tags:", soup.a)
print("\n")
print("Get all HTML reference tags:", soup.find_all('a'))
print("\n")
print("Get each HTML reference tag:")
for ref_tag in soup.find_all('a'):
    print(ref_tag)
print("Get the class assigned to a specific HTML element:", soup.a['class'])
print("\n")
print("Get a specific HTML tag by ID:", soup.find(id="navbarDropdown3"))
print("\n")

# # As shown above, each HTML tag found in a page is created as a property of
# the soup.  The property text includes the tags
footer_text = soup.footer
print("Footer: ", footer_text)
print("\n")

# We can get specific elements within the property using find_all.  A list
# is returned containing all matches.  The search is recursive.
ref_in_footer = soup.footer.find_all('a')
print("References in the footer:", ref_in_footer)
print("\n")

# We can pass a list of elements to return more than one type of tag.  The
# search is recursive, so you may find <a> within <p> text, and a later
# returned element will be just the <a> tag that was in the <p> tag.
print("Paragraph and reference elements in the body:")
print("-----------------------------------------------")
for element in soup.body.find_all(['p','a']):
    print(element, "\n")
    
print("-----------------------------------------------")
print("\n")

# We can get specific attributes by tag using the get().  If there are more than
# one paragraph tag, can use find_all and iterate through each one to get the
# style.
p_style_in_footer = soup.footer.p.get('style')
print("Paragraph style in footer:", p_style_in_footer)
print("\n")


# We can get all URL's from a webpage
print("Get all URL's on the page, including relative:")
for link in soup.find_all('a'):
    print(link.get('href'))

