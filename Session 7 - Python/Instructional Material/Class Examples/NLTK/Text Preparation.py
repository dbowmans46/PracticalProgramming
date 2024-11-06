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
# print("English Stopwords:")
# print(nltk.corpus.stopwords.words('english'))
# print("\n\n\n")

# # As is alluded to, stopwords for multiple languages are included
# print("Russian Stopwords:")
# print(nltk.corpus.stopwords.words('russian'))
# print("\n\n\n")

# print("Hindi Stopwords (using Latin characters and borrowed English):")
# print(nltk.corpus.stopwords.words('hinglish'))
# print("\n\n\n")

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



# raw_text = "He does run the store like a drill seargent."    
# tokenized_text = nltk.word_tokenize(raw_text)
# pos_text1 = nltk.pos_tag(tokenized_text)
# print(pos_text1,"\n")

# # Notice that the part of speech changes depending on how the word is used.  In
# # this example, compare the word "run"
# raw_text2 = "The run was not as good as the previous three."    
# tokenized_text2 = nltk.word_tokenize(raw_text2)
# pos_text2 = nltk.pos_tag(tokenized_text2)
# print(pos_text2)

# # To get all words with a specific part of speech, we can use a loop or a list
# # comprehension.  We will get adverbs in this example (POS token is RB)
# RB_words = []
# for word_pos_pair in pos_text2:
#     if word_pos_pair[1] == 'RB':
#         RB_words.append(word_pos_pair[0])
        
# RB_words = [words[0] for words in pos_text2 if words[1] == 'RB']
# print(RB_words)

# # Getting a list of the parts of speach from a pre-trained text
# pos_dict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
# for key_val in pos_dict:
#     print(key_val, ":", pos_dict[key_val], "\n")



###############################################################################
#                                                                             #
#                                  Stemming                                   #
#                                                                             #
###############################################################################


# # Create a stemmer and use on some test text
# example_text = "He studying way too hard on maliciously and unnecessarily assigned reading assignments from uncaring professors."
# tokenized_text = nltk.word_tokenize(example_text)
# porterStemmer = nltk.stem.PorterStemmer()
# for token in tokenized_text:
#     print("Token:", token, ", Stem:", porterStemmer.stem(token))
    
# # If we look at one of the previous sentences, can we see any issues with
# # the stem produced?
# raw_text = "He does run the store like a drill seargent."    
# tokenized_text = nltk.word_tokenize(raw_text)
# for token in tokenized_text:
#     print("Token:", token, ", Stem:", porterStemmer.stem(token))

# # Let's see if a different stemmer may do a better job
# print("\n")
# lancasterStemmer = nltk.stem.LancasterStemmer()
# for token in tokenized_text:
#     print("Token:", token, ", Stem:", lancasterStemmer.stem(token))




###############################################################################
#                                                                             #
#                               Lemmatization                                 #
#                                                                             #
###############################################################################

# from nltk.stem import WordNetLemmatizer as wnl

# # The lemmatize function takes two arguments: word to lemmaize and the part of 
# # speech.  If a part of speech isn't given, it will default to a noun.  If it
# # cannot find a matching word and part of speech, it will return the given word
# # input.  The possible parts of speech are as follows:
# # “n” -- nouns
# # “v” -- verbs
# # “a” -- adjectives
# # “r” -- adverbs
# # “s” -- satellite adjectives, which appear next to the word it describes, 
# #        i.e. blue car
# print("Lemmatized 'was':", wnl().lemmatize('was', 'v'))

# # If we want to use the part-of-speech tagging from the built-in tagger with
# # NLTK, we want to use the first character as the major classification for
# # the part of speech.
# example_text = "He studying way too hard on maliciously and unnecessarily assigned reading assignments from uncaring professors."
# tokenized_text = nltk.word_tokenize(example_text)
# pos_tagged_text = nltk.pos_tag(tokenized_text)
# lemmas = []
# for word,pos in pos_tagged_text:
#     if pos[0] in ["N", "V", "A", "R"]:
#         # If we can normalize the part-of-speech based on the 1st character of
#         # the word tree POS, do so
#         wnl_pos = pos[0].lower()
#     else:
#         # Default to noun, the default of the WordNetLemmatizer
#         wnl_pos = "n"
    
#     lemmas.append(wnl().lemmatize(word, wnl_pos))
    
# print(lemmas)
