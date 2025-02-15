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
#     if word.lower() not in nltk.corpus.stopwords.words('english'):
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



# raw_text = "He does run the store like a drill sergeant."    
# tokenized_text = nltk.word_tokenize(raw_text)
# pos_text1 = nltk.pos_tag(tokenized_text)
# print(pos_text1,"\n")

# # Notice that the part of speech changes depending on how the word is used.  In
# # this example, compare the word "run"
# raw_text2 = "The fourth night run was not as good as the previous three."    
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

# # Getting the descriptions of a list of the parts of speach from a pre-trained text
# pos_dict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
# for key_val in pos_dict:
#     print(key_val, ":", pos_dict[key_val], "\n")



###############################################################################
#                                                                             #
#                                  Stemming                                   #
#                                                                             #
###############################################################################


# # Create a stemmer and use on some test text
# example_text = "He is studying way too hard on maliciously and unnecessarily assigned reading assignments from uncaring professors."
# tokenized_text = nltk.word_tokenize(example_text)
# porterStemmer = nltk.stem.PorterStemmer()
# for token in tokenized_text:
#     print("Token:", token, ", Stem:", porterStemmer.stem(token))

# print("\n")

# # Let's create a side-by-side comparison of the Porter and Lancaster stemmers
# #raw_text = "He does run the store like a drill seargent."   
# raw_text = "He studied way too hard on maliciously and unnecessarily assigned reading assignments from uncaring professors." 
# tokenized_text = nltk.word_tokenize(raw_text)
# porterStemmer = nltk.stem.PorterStemmer()
# lancasterStemmer = nltk.stem.LancasterStemmer()
# porter_stems = []
# lancaster_stems = []
# for token in tokenized_text:
#     porter_stems.append(porterStemmer.stem(token))
#     lancaster_stems.append(lancasterStemmer.stem(token))

# # Let's put all this in a DataFrame for a clear comparison
# import pandas as pd

# df = pd.DataFrame([tokenized_text, porter_stems, lancaster_stems])
# df = df.transpose()
# df.columns=["Tokens", "Porter", "Lancaster"]
# print(df)
    
# # Notice the more-heavily truncated stems of the lancaster stemmer vs the porter
# # stemmer.



# # Finding stemmers and token of the longest word
# longest_word=""
# for rows in range(df.shape[0]):
#     if len(longest_word) < len(df.iloc[rows]['Tokens']):
#         longest_word = df.iloc[rows]['Tokens']

# print("Token:", longest_word)
# print("Porter:", porterStemmer.stem(longest_word))
# print("Lancaster:", lancasterStemmer.stem(longest_word))



# # Finding the same stems of different words
# import pandas as pd
# dracula_filepath = "/home/doug/repos/PracticalProgramming/Session 7 - Python/Instructional Material/In-Class Exercises/Books/Dracula.txt"
# with open(dracula_filepath) as fileHandle:
#     dracula_string = fileHandle.read()
# dracula_tokens = nltk.word_tokenize(dracula_string)
# draculaText = nltk.Text(dracula_tokens)
# porterStemmer = nltk.stem.PorterStemmer()
# lancasterStemmer = nltk.stem.LancasterStemmer()
# porter_stems = []
# lancaster_stems = []
# context = []
# index=0
# for token in dracula_tokens:
#     porter_stems.append(porterStemmer.stem(token))
#     lancaster_stems.append(lancasterStemmer.stem(token))
    
#     if index < 5:
#         context.append(" ".join(dracula_tokens[0:index+5]))
#     elif index > len(dracula_tokens) - 5:
#         context.append(" ".join(dracula_tokens[index-5:len(dracula_tokens)]))
#     else:
#         context.append(" ".join(dracula_tokens[index-5:index+5]))
        
#     index += 1

# dracula_df = pd.DataFrame([dracula_tokens, porter_stems, lancaster_stems, context])
# dracula_df = dracula_df.transpose()
# dracula_df.columns=["Tokens", "Porter", "Lancaster", "Context"]
# print(dracula_df)

# # Can use this to find some examples that may be useful for finding different
# # words
# dracula_df_mismatch = dracula_df[dracula_df["Porter"] != dracula_df["Lancaster"]]

# # Then we can look up all the entries for this in a stemmer to see if there
# # are different tokens
# dracula_df_mad = dracula_df[dracula_df["Lancaster"] == "mad"]

# tokens_of_mad = []
# for token_val in dracula_df_mad["Tokens"]:
#     if token_val not in tokens_of_mad:
#         tokens_of_mad.append(token_val)

# If we want to automate this, we could check each stem in one of the Porter and
# /or Lancaster stemmers.  For each of these stems, find all associated tokens
# and see if there are different words in this list.


# Another way to do this would be to group by stem, and find the stem that has
# the highest count of associated tokens (rows).






###############################################################################
#                                                                             #
#                               Lemmatization                                 #
#                                                                             #
###############################################################################

# from nltk.stem import WordNetLemmatizer as wnl
# 
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
# 
# # If we want to use the part-of-speech tagging from the built-in tagger with
# # NLTK, we want to use the first character as the major classification for
# # the part of speech.
# example_text = "He was studying way too hard on maliciously and unnecessarily assigned reading assignments from uncaring professors despite the obvious signs of exhaustion slowly invading his faculties."
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
#     
#     lemmas.append(wnl().lemmatize(word, wnl_pos))
#     
# print(lemmas)
