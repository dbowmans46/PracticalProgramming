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

from WWCardsDeck import WWCardsDeck
from WWInitialDeck import WWInitialDeck
from WWShuffleDeck import WWShuffleDeck
"""
@ brief Game Manager is responsible for place holding properties
"""


class WWGameManager:

    '''
    @ brief Dealing deck, used to create all players decks
    @ param str-list
    '''
    gameDeck = WWCardsDeck([])

    '''
     @ brief Deck holding all wagers
     @ param WWCardsDeck
     '''
    wagerDeck = WWCardsDeck([])

    '''
     @ brief Deck that computer plays from
     @ param WWCardsDeck
     '''
    computerDeck = WWCardsDeck([])

    '''
     @ brief Computers winnings, After each turn, will be used again
     @ param WWCardsDeck
     '''
    computerGraveyardDeck = WWShuffleDeck([])

    '''
     @ brief computer's card being used in war. does not include wagers
     @ param WWCardsDeck
     '''
    computerBattleDeck = WWCardsDeck([])
    '''
     @ brief Deck that player plays from
     @ param WWCardsDeck
     '''
    playerDeck = WWCardsDeck([])

    '''
     @ brief players winnings, After each turn, will be used again
     @ param WWCardsDeck
     '''
    playerGraveyardDeck = WWShuffleDeck([])

    '''
     @ brief player's card being used in war. does not include wagers
     @ param WWCardsDeck
     '''
    playerBattleDeck = WWCardsDeck([])
    '''
     @ brief Name of player from WWSetupWindow
     '''
    playerName = 'PLAYER'

    '''
     @ brief Name of computer from WWSetupWindow
     '''
    compName = 'COMPUTER'

    '''
     @ brief User input from WWSetupWindow sets number decks
     '''
    deckCount = 1

    '''
     @ brief User input from WWSetupWindow sets deck style 
     @ details deck color input box
     '''

    deckStyle = ''

    '''
     @ brief Number of turns completed
     '''
    turnCount = 0

    '''
     @ brief Name variable used at WWVictoryWindow to show who wins
     '''
    winnerName = ''
    '''
     @ number of Wars in a row
     '''
    warCount = 0
    
    '''
    @ Switch to tell the game to restart if the player chooses
    '''
    playAgainToggle = True
    
