#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:24:05 2024

@author: doug
"""

import nltk

# First, we must get the raw text inside python
dracula_filepath = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Books/Dracula.txt"
with open(dracula_filepath) as fileHandle:
    dracula_string = fileHandle.read()
    
# Next, we can convert the raw text into tokens, and then into an NLTK.Text 
# object
dracula_tokens = nltk.word_tokenize(dracula_string)
draculaText = nltk.Text(dracula_tokens)

# We can then do our basic assessment
print("Count of 'vampire': ", draculaText.count("vampire"))
print("Context of 'vampire': ", draculaText.concordance("vampire"))
print("20 Most common words: ", draculaText.vocab().most_common(20))
