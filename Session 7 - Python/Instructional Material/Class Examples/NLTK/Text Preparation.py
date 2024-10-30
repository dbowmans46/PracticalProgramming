#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:44:49 2024

@author: doug
"""

import nltk

###############################################################################
#                                                                             #
#                                Tokenization                                 #
#                                                                             #
###############################################################################

# # First, we must get the raw text inside python
# dracula_filepath = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Books/Dracula.txt"
# with open(dracula_filepath) as fileHandle:
#     dracula_string = fileHandle.read()
    
# # Next, we can convert the raw text into tokens, and then into an NLTK.Text 
# # object
# dracula_tokens = nltk.word_tokenize(dracula_string)
# draculaText = nltk.Text(dracula_tokens)


# # We can then do any basic assessment.  Note that .concordance() will print
# # out the context of the words, but returns None, and thus cannot be used
# # in the print statement.
# print("Count of 'vampire': ", draculaText.count("vampire"))
# print("\n\n\n")
# print("Context of 'vampire':")
# draculaText.concordance("vampire")
# print("\n\n\n")
# print("Context of 'carriage':")
# draculaText.concordance("carriage")
# print("20 Most common tokens: ", draculaText.vocab().most_common(20))



###############################################################################
#                                                                             #
#                             Removing Stop Words                             #
#                                                                             #
###############################################################################

# from nltk.book import *

# # Stopwords: nltk.corpus.stopwords.words('english')
# nltk.corpus.stopwords.words('english')

# # As is alluded to, stopwords for multiple languages are included
# nltk.corpus.stopwords.words('russian')
# nltk.corpus.stopwords.words('hinglish')

# # Removing stop words
# # There are many ways to do this, one of which is using a list comprehension
# meaningful_words = [moby_dick_words for moby_dick_words in text1.tokens if not moby_dick_words.lower() in nltk.corpus.stopwords.words('english')]

# # Another way is to use a for loop
# meaningful_words_from_loop = []
# for word in text1.tokens:
#     if word not in nltk.corpus.stopwords.words('english'):
#         meaningful_words_from_loop.append(word)

# # Removing punctuation and articles.  Be careful with this one: punctuation 
# # and/or articles may be useful in some cases.
# meaningful_words = [no_punc_words for no_punc_words in meaningful_words if len(no_punc_words) > 2]
# freq_distribution = nltk.FreqDist(meaningful_words)
# for word in freq_distribution:
#     print(word, ":", freq_distribution[word])
    




###############################################################################
#                                                                             #
#                            Part of Speech Tagging                           #
#                                                                             #
###############################################################################



raw_text = "He does run the store like a drill seargent."    
tokenized_text = nltk.word_tokenize(raw_text)
pos_text1 = nltk.pos_tag(tokenized_text)
print(pos_text1,"\n")

# Notice that the part of speech changes depending on how the word is used.  In
# this example, compare the word "run"
raw_text2 = "The run was not as good as the previous three."    
tokenized_text2 = nltk.word_tokenize(raw_text2)
pos_text2 = nltk.pos_tag(tokenized_text2)
print(pos_text2)

# # Getting a list of the parts of speach from a pre-trained text
# pos_dict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
# for key_val in pos_dict:
#     print(key_val, ":", pos_dict[key_val], "\n")



###############################################################################
#                                                                             #
#                                  Stemming                                   #
#                                                                             #
###############################################################################


# Create a stemmer and use on some test text
example_text = "He was studying way to hard on malicsiouly assigned reading assignments."
tokenized_text = nltk.word_tokenize(example_text)
porterStemmer = nltk.stem.PorterStemmer()
for token in tokenized_text:
    print("Token:", token, ", Stem:", porterStemmer.stem(token))
    
# If we look at one of the previous sentences, can we see any issues with
# the stem produced?
raw_text = "He does run the store like a drill seargent."    
tokenized_text = nltk.word_tokenize(raw_text)
for token in tokenized_text:
    print("Token:", token, ", Stem:", porterStemmer.stem(token))

# Let's see if a different stemmer may do a better job
print("\n")
lancasterStemmer = nltk.stem.LancasterStemmer()
for token in tokenized_text:
    print("Token:", token, ", Stem:", lancasterStemmer.stem(token))