# -*- coding: utf-8 -*-
"""
LICENSE (MIT License):

Copyright 2018 Jason Gilbert, Ryan Concienne, and Douglas Bowman

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

"""
@ brief Parent cardsDeck
"""


class WWCardsDeck():
    """
    @ brief Constructor setup
    @ param string-list cards strings are filenames of card images
    """

    def __init__(self, cards):
        self.cards = cards

    """
    @ brief Remove the top card from cards list and adds it to the end of destDeck list.
    @ param WWCardsDeck destDeck The destination deck to which the card is transferred. 
    """

    def cardTransfer(self, destDeck):
        if len(self.cards) > 0:
            topCard = self.cards.pop()
            destDeck.cards.append(topCard)

    """
    @ brief Transfer all the cards in the deck to destination deck.
    @ param WWCardsDeck destDeck The destination deck to which the deck is transferred. 
    """

    def cardTransferAll(self, destDeck):
        destDeck.cards.extend(self.cards)
        self.cards = []

    """
    @ brief Remove the first card from cards list and adds it to the end of destDeck list.
    @ param WWCardsDeck destDeck The destination deck to which the card is transferred. 
    """

    def cardTransferALT(self, destDeck):
        if len(self.cards) > 0:
            topCard = self.cards.pop(0)
            destDeck.cards.append(topCard)
