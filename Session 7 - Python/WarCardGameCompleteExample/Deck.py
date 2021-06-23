#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:41:30 2018

@author: Douglas Bowman

LICENSE (MIT License):

Copyright 2018 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""

import random
from WarConstants import WarConstants

"""
@brief Class that houses a deck of standard poker playing cards, possibly comprising 
       of multiple single standard decks (i.e. there can be repeat cards from a 
       standard deck of 52 cards)
"""
class Deck():    
    
    """
    @brief constructor, set default values
    @param cardCount The number of cards in the deck.  Can be more than 52
    """
    def __init__(self, deckCount):
        self.DeckCount = deckCount
        self._CreateDeck()
        
    
    """
    @brief Create the initial deck of cards
    """
    def _CreateDeck(self):
        
        self.Cards = []
        
        # Copy the standard 52 card deck enough times to fill the War deck
        for Decks in range(self.DeckCount):
            self.Cards.extend(WarConstants.CARD_FILENAMES)
    
    """
    @brief Randomizes the deck in place
    """
    def Shuffle(self):
        # Create the number of indices for the cards in a list.
        # Randomly pick an indice for this list of indices, add that card to
        # the actual playing card list, and remove the indice of the indice.
        # Can also use the random.shuffle() STL method
        
        tempCards = self.Cards
        randDeck = []
        while len(tempCards) > 0:
            remIndex = random.randint(0,len(tempCards)-1)
            randDeck.append(tempCards[remIndex])
            del tempCards[remIndex]
            
        self.Cards = randDeck
        
    """
    @brief Cuts the deck in two, and outputs the two decks as individual decks
    @returns The current deck cut into two decks by dividing at the middle index
    """
    def SplitDeck(self):
        
        return(self.Cards[0:int(len(self.Cards)/2)], self.Cards[int(len(self.Cards)/2):int(len(self.Cards))])
            
        