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

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QApplication
from WWSetupWindow import WWSetupWindow
from WWWarConstants import WWWarConstants
from WWGameManager import WWGameManager
from WWGameWindow import WWGameWindow
from WWVictoryWindow import WWVictoryWindow
from WWDataLogger import WWDataLogger
from WWCardsDeck import WWCardsDeck
from WWShuffleDeck import WWShuffleDeck


if __name__ == "__main__":
    wwgm = WWGameManager()
    const = WWWarConstants()
    app = QApplication(sys.argv)
    # this will delete the "game_log.json" before running new game
    WWDataLogger.deleteLogger()

    while (WWGameManager.playAgainToggle):
        print("NEW GAME")
        #wwgm = WWGameManager()
        print("PB over- ", len(WWGameManager.playerBattleDeck.cards))
        print("CB over- ", len(WWGameManager.computerBattleDeck.cards))
        print("Winner over- ", WWGameManager.winnerName)
        # Turn off right away to prevent possible infinite loops
        WWGameManager.playAgainToggle = False

        wwsw = WWSetupWindow()
        wwsw.setupUi()
        while wwsw.wwswIsActive == True:
            QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.exit()
        wwsw.Dialog.close()

        wwgw = WWGameWindow()
        wwgw.setTheStage()
        while wwgw.wwgwIsActive == True:
            QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.exit()
        wwgw.MainWindow.close()

        wwvw = WWVictoryWindow()
        wwvw.setTheStage()
        while wwvw.wwvwIsActive == True:
            QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.exit()
        wwvw.MainWindow.close()

        # Reset game manager values when playing again
        if WWGameManager.playAgainToggle:

            '''
            @ brief Dealing deck, used to create all players decks
            @ param str-list
            '''
            WWGameManager.gameDeck = WWCardsDeck([])

            '''
             @ brief Deck holding all wagers
             @ param WWCardsDeck
             '''
            WWGameManager.wagerDeck = WWCardsDeck([])

            '''
             @ brief Deck that computer plays from
             @ param WWCardsDeck
             '''
            WWGameManager.computerDeck = WWCardsDeck([])

            '''
             @ brief Computers winnings, After each turn, will be used again
             @ param WWCardsDeck
             '''
            WWGameManager.computerGraveyardDeck = WWShuffleDeck([])

            '''
             @ brief computer's card being used in war. does not include wagers
             @ param WWCardsDeck
             '''
            WWGameManager.computerBattleDeck = WWCardsDeck([])
            '''
             @ brief Deck that player plays from
             @ param WWCardsDeck
             '''
            WWGameManager.playerDeck = WWCardsDeck([])

            '''
             @ brief players winnings, After each turn, will be used again
             @ param WWCardsDeck
             '''
            WWGameManager.playerGraveyardDeck = WWShuffleDeck([])

            '''
             @ brief player's card being used in war. does not include wagers
             @ param WWCardsDeck
             '''
            WWGameManager.playerBattleDeck = WWCardsDeck([])

            WWGameManager.winnerName = ""
            WWGameManager.warCount = 0
            #del wwgm
            #print("Before del - wwgm", wwgm)
            #wwgm = WWGameManager()
            #print("After del - wwgm", wwgm)
            del wwsw
            del wwgw
            del wwvw
            print("PB - del", len(WWGameManager.playerBattleDeck.cards))
            print("CB - del", len(WWGameManager.computerBattleDeck.cards))
            print("Winner - del", WWGameManager.winnerName)
            print("DC", WWGameManager.deckCount)
    sys.exit()
    # sys.exit(app.exec_())
# TODO print wwgm-referance types (objects itself)

"""
Overlord.py is the game executable *is WagerWar*
overlord runs game manager
overlord opens setup window
(setup window waiting for user inputs)
(setup window pass info to game manager)
(setup window closes)
overlord opens game window 
(game window grabs info from game manager)
(game window waiting for user inputs/and updating game manager)
(in Vic/Defeat scenario, game window closes)
Overlord opens Victory Window
(Game Manager passes info to Victory Screen)
"""
