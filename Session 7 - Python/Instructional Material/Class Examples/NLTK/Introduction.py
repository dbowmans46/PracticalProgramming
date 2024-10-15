#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:45:55 2021

Copyright 2024 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import nltk

# Get corpuses. When promp, (d), all to get everything.  Otherwise, can 
# download just what you want
nltk.download()

# TODO: Import all the included texts using from nltk.book import *
# TODO: Import other texts, such as nltk.corpus.pros_cons
# TODO: Look at texts that have part-of-speech tags, such as nltk.corpus.brown.tagged_words()
# TODO: use text1.concordance("monstrous") to show words with context
# TODO: Get the concordance of other words
# TODO: Calculate statistical values of text
# TODO: Count of word: text1.count(<word>)
# TODO: Frequency distribution of words/symbols: nltk.probability.FreqDist(<text_to_analyze>) or <text_to_analyze>.vocab()
# TODO: Get similar words with text1.similar("monstrous")
# TODO: text3.generate() to generate text based on a corpus
# TODO: Frequency distribution of words nltk.FreqDist(text1)

# Context

####
# Text Preparation
####

# Stopwords: nltk.corpus.stopwords.words('english')
# As is alluded to, stopwords for multiple languages are included
#nltk.corpus.stopwords.words('russian')
#nltk.corpus.stopwords.words('hinglish')
# Removing stop words
#meaningful_words = [moby_dick_words for moby_dick_words in text1.tokens if not lower(moby_dick_words) in nltk.corpus.stopwords.words('english')]

# Removing punctuation and articles
#meaningful_words = [no_punc_words for no_punc_words in meaningful_words if len(no_punc_words) > 2]
#nltk.FreqDist(meaningful_words)


# Distances between words
#Jaccard Distance
#Damerau-Levenshtein Distance