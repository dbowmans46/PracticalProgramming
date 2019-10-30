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

#from random import shuffle

"""
@ brief Parent cardsDeck
@ param cards is a list of cards
"""
class WWCardsDeck():
    """
    @ brief Constructor setup
    @ param string-list cards 
    """
    def __init__(self, cards):
        self.cards = cards
        


    """..
    @ brief Remove the top card from cards list and adds it to the end of destDeck list.
    @ param WWCardsDeck destDeck The destination deck to which the card is transferred. 
    """
    def cardTransfer(self, destDeck):
        # removes last card from list of cards moves it to destDeck
        topCard = self.cards.pop()
        destDeck.cards.append(topCard)
        
        
#orgDeck = ['c01.bmp','c02.bmp','c03.bmp',                        'c04.bmp',


 #                       'c05.bmp']


#destDeck = ['d01.bmp',
#                        'd02.bmp',
#                        'd03.bmp',
#                        'd04.bmp',
#                        'd05.bmp',
#                        'd06.bmp']
#oldDeck = WWCardsDeck(orgDeck)
#newDeck = WWCardsDeck(destDeck)
