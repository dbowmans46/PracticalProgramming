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

from WWDeck import WWDeck
"""
@ brief Game Manager is responsible for place holding properties
"""
class WWGameManager:
        
     '''
     @ brief Dealing deck, used to create all players decks
     '''
     gameDeck = WWDeck()
     
     '''
     @ brief Deck that computer plays from
     '''
     computerDeck = WWDeck()
     
     '''
     @ brief Computers winnings, After each turn, will be used again
     '''
     computerDiscardDeck = WWDeck()
     
     '''
     @ brief Deck that player plays from
     '''
     playerDeck = WWDeck()
     
     '''
     @ brief players winnings, After each turn, will be used again
     '''
     playerDiscardDeck = WWDeck()
    
     '''
     @ brief Name of player from WWSetupWindow
     '''
     playerName = 'a'
     
     '''
     @ brief Name of computer from WWSetupWindow
     '''
     compName = 'b'
     
     '''
     @ brief User input from WWSetupWindow sets number decks
     '''
     deckCount = 1
     
     '''
     @ brief User input from WWSetupWindow sets deck style
     '''
     #deck color input box
     deckStyle = 'c'
     
     '''
     @ brief Number of turns completed
     '''
     turnCount = ''
     
     '''
     @ brief Name variable used at WWVictoryWindow to show who wins
     '''
     winnerName = ''
     
     
         
         
'''
This Window will remain passive. Basically just a placeholder for info

'''