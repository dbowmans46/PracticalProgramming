#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:20:04 2018

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

import time

from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore

from PyQt5 import QtGui

from WarConstants import WarConstants
from Deck import Deck
from GameManager import GameManager
from GameManager import Challenger

class GameWindow(QWidget):    
    
    """
    @brief Constructor, set default parameters
    @param int left Distance of left side of window from left side of monitor screen
    @param int top Distance of top side of window from top of monitor screen
    @param int width The width of the window in pixels
    @param int height The height of the window in pixels
    @param string windowTitle The title of the setup window
    @param GameManager gameMan The game manager for the game
    """
    def __init__(self, left, top, width, height, windowTitle, gameMan):
        super().__init__()
        self.setGeometry(left, top, width, height)
        self.setWindowTitle(windowTitle)
        self.GameMan = gameMan
        
        self.CardCount = self.GameMan.DeckCount * WarConstants.DECK_SIZE
        
        # Setup some geometry variables
        # The space between the window edge border and the content
        self.EdgeBuffer = 20
        self.LabelWidth = self.geometry().width()/2 - self.EdgeBuffer
        self.LabelHeight = 40
        
        # list to hold all of the cards that will be added to the winner's hand
        self.SpoilsOfWarList = []
        
        self.setMouseTracking = True
        self.Winner = ""
        
        self._LayoutSetup()
        
        self._GameUpdateLoop()
        
        
    """
    @brief Create the window layout
    """
    def _LayoutSetup(self):
        # Create the label for the player/comp score, on the left of the screen
        self.PlayerScoreLabel = QLabel(self.GameMan.PlayerName,self)
        self.PlayerScoreLabel.setGeometry(self.EdgeBuffer, self.EdgeBuffer, 
                                          self.LabelWidth, self.LabelHeight)
        
        self.ComputerScoreLabel = QLabel(self.GameMan.CompName,self)
        self.ComputerScoreLabel.setGeometry(self.geometry().width()/2, 
                                            self.EdgeBuffer, self.LabelWidth, 
                                            self.LabelHeight)
        
        # Add the values to the labels
        self.PlayerScoreValLabel = QLabel(str(self.CardCount/2),self)
        self.PlayerScoreValLabel.setGeometry(self.EdgeBuffer, 
                                        self.PlayerScoreLabel.geometry().top() + self.EdgeBuffer, 
                                        self.LabelWidth, self.LabelHeight) 
        
        self.CompScoreValLabel = QLabel(str(self.CardCount/2),self)
        self.CompScoreValLabel.setGeometry(self.geometry().width()/2, 
                                      self.PlayerScoreLabel.geometry().top() + self.EdgeBuffer, 
                                      self.LabelWidth, self.LabelHeight)
        
        # Create the deck location.  
        self.PlayerCardLoc = QLabel(self)
        self.PlayerCardPix = QPixmap(WarConstants.CARD_PIC_LOC + 'blueBackVert.bmp')
        self.PlayerCardLoc.setPixmap(self.PlayerCardPix)
        self.PlayerCardLoc.resize(self.PlayerCardPix.width(), self.PlayerCardPix.height())
        self.PlayerCardLoc.move(WarConstants.BORDER_WIDTH_BUFFER_SIZE, 
                                self.geometry().height()/2-self.PlayerCardLoc.geometry().height()/2)
        
        # Create the drawn card for the player.  Don't show until after the button
        # is pressed
        self.PlayerDrawnCard = QLabel(self)
        self.PlayerDrawnCard.resize(self.PlayerCardPix.width(), self.PlayerCardPix.height())
        self.PlayerDrawnCard.move(2*WarConstants.BORDER_WIDTH_BUFFER_SIZE + self.PlayerDrawnCard.width(), 
                                     self.geometry().height()/2-self.PlayerDrawnCard.geometry().height()/2)
        
        # Create the computer deck location
        self.CompCardLoc = QLabel(self)
        self.CompCardPix = QPixmap(WarConstants.CARD_PIC_LOC + 'blueBackVert.bmp')
        self.CompCardLoc.setPixmap(self.CompCardPix)
        self.CompCardLoc.resize(self.CompCardPix.width(), self.CompCardPix.height())
        self.CompCardLoc.move(self.width() - WarConstants.BORDER_WIDTH_BUFFER_SIZE - self.CompCardLoc.width(), 
                              self.height()/2-self.CompCardLoc.height()/2)
        
        # Create the drawn card for the player.  Don't show until after the button
        # is pressed
        self.CompDrawnCard = QLabel(self)
        self.CompDrawnCard.resize(self.CompCardPix.width(), self.CompCardPix.height())
        self.CompDrawnCard.move(self.width() - 2*WarConstants.BORDER_WIDTH_BUFFER_SIZE - 2*self.CompDrawnCard.width(), 
                                self.height()/2-self.CompDrawnCard.height()/2)
        
        # Add the deck of cards, shuffle, and distribute
        self.GameDeck = Deck(self.GameMan.DeckCount)
        self.GameDeck.Shuffle()        
        (self.PlayerCards,self.CompCards) = self.GameDeck.SplitDeck()
        self.GameMan.PlayerCardCount = len(self.PlayerCards)
        self.GameMan.CompCardCount = len(self.CompCards)
        
        
        # Add buttons to display the cards
        self.PlayerCardButton = QPushButton(self)
        self.PlayerCardButton.setText("Draw Player Card")
        self.PlayerCardButton.clicked.connect(self._PlayerCardClick)
        self.PlayerCardButton.move(WarConstants.BORDER_WIDTH_BUFFER_SIZE, 
                                   self.height() - self.PlayerCardButton.height() - WarConstants.BORDER_HEIGHT_BUFFER_SIZE)
        
        # Add buttons to run through the entire game without further user input
        self.CompleteGameButton = QPushButton(self)
        self.CompleteGameButton.setText("Auto-complete Game")
        self.CompleteGameButton.clicked.connect(self._CompleteGameClick)
        self.CompleteGameButton.move(2*WarConstants.BORDER_WIDTH_BUFFER_SIZE + self.PlayerCardButton.width(), 
                                   self.height() - self.CompleteGameButton.height() - WarConstants.BORDER_HEIGHT_BUFFER_SIZE)
        
        # Add quit button
        self.QuitButton = QPushButton(self)
        self.QuitButton.setText('Quit')
        self.QuitButton.clicked.connect(self._QuitButtonClick)
        self.QuitButton.move(self.width() - self.QuitButton.width() - WarConstants.BORDER_WIDTH_BUFFER_SIZE,
                                self.height() - self.QuitButton.height() - WarConstants.BORDER_HEIGHT_BUFFER_SIZE)
        
        # Create boolean to tell if player wants to quit
        self.Quit = False
        
        self.show()
        
    """
    #brief Ensure that the player cards and comp cards sum to the total number of cards
    @param list addList An additional list to add to the total card count
    """
    def _CheckCards(self, addList=[]):

        totalCards = len(self.PlayerCards) + len(self.CompCards) + len(addList)
        if (totalCards != self.GameMan.DeckCount*WarConstants.DECK_SIZE):
            raise ValueError("Total number of cards does not equal the total deck size")
        else:
            print("Card check OK")
            
        return
    
    """
    @brief Get the count of the cards
    @param list addList An additional list to add to the total card count
    """
    def _PrintCardCounts(self, addList=[]):
        print("Player card count: " + str(len(self.PlayerCards)))        
        print("Comp card count: " + str(len(self.CompCards)))
        print("Additional card count: " + str(len(addList)))
    
        return
    
    """
    @brief Setup the main loop that will wait for player inputs and update the
           game accordingly
    """
    def _GameUpdateLoop(self):
        
        gameOver = False
        
        while  ( (gameOver == False) and (self.Quit == False) ):
            
            # Process events and wait for user input
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.05)
            
            #self._CheckCards()
            
            # Check if one player has all of the cards. if so, the game is over
            if ( (len(self.PlayerCards) == 0) or (len(self.CompCards) ==0) ):
                gameOver = True
                print("----------------------------------------------------")
                print("End Game Values:")
                print("++++++++++++++++")
                print("gameOver: " + str(gameOver))
                print("Deck Size: " + str(WarConstants.DECK_SIZE*self.GameMan.DeckCount))
                self._PrintCardCounts(self.SpoilsOfWarList)                
                print("----------------------------------------------------")
                
                #TODO: Print scores and winner
                if (len(self.PlayerCards) == WarConstants.DECK_SIZE*self.GameMan.DeckCount):
                    self.Winner = self.GameMan.PlayerName
                    print(self.Winner + " Won")
                elif (len(self.CompCards) == WarConstants.DECK_SIZE*self.GameMan.DeckCount):
                    self.Winner = self.GameMan.CompName
                    print(self.Winner + " Won")
                    
            self._DrawGraphics()                    

        return
    
    """
    @brief Run through the entire game without user input until the end
    """
    def _CompleteGameClick(self):
        
        while (len(self.PlayerCards) > 0 and (len(self.CompCards) > 0) ):
            self._PlayerCardClick()
        
        return
    
    """
    @brief Slot for player button pressed.  Show new cards and calculate new 
           scores/deck sizes
    """
    def _PlayerCardClick(self):
        
        # TODO: Calculate new scores and update the gameMan appropriately
        if ( (len(self.PlayerCards) > 0) and (len(self.CompCards) > 0) ):
            self._CalculateScore()
            
            #self.SpoilsOfWarList = []
        
        return
    
    """
    @brief Slot for quit button, set quit property
    """
    def _QuitButtonClick(self):
        self.Quit = True
    
    """
    @brief Compare cards drawn and calculate new scores
    @param int cardToCompare
    @retval string The player that won the battle
    """
    def _CalculateScore(self, cardToCompare=0):
        
        try:
            playerCardVal = WarConstants.CARD_VALUES[self.PlayerCards[cardToCompare]]
            compCardVal = WarConstants.CARD_VALUES[self.CompCards[cardToCompare]]
            
            print("playerCardVal: " + str(playerCardVal))
            print("compCardVal: " + str(compCardVal))
        except:
            print("Error Check for _CalculateScore:")
            print("cardToCompare:" + str(cardToCompare))
            self._PrintCardCounts(self.SpoilsOfWarList)
            print(self.PlayerCards[cardToCompare])
            print(self.CompCards[cardToCompare])
            print("")
            
        #print("Player card value: " + str(playerCardVal))
        #print("Computer card value: " + str(compCardVal))
        self._PrintCardCounts()
        #self._CheckCards()
        
        winner = ""
        
        if (playerCardVal > compCardVal):
            self.PlayerCards.append(self.CompCards.pop(cardToCompare))
            self._MoveCardToBack(self.PlayerCards)
            winner = Challenger.PLAYER
            #self._CheckCards()
            
        elif (playerCardVal < compCardVal):
            self.CompCards.append(self.PlayerCards.pop(cardToCompare))
            self._MoveCardToBack(self.CompCards)
            winner = Challenger.COMPUTER
            #self._CheckCards()
            
        elif (playerCardVal == compCardVal):
            #self._WarDraw2()   #works
            self._WarDraw()
            print("\n______________________________________")
            print("Post War Check:")
            self._PrintCardCounts(self.SpoilsOfWarList)
            totalCards = len(self.PlayerCards) + len(self.CompCards) + len(self.SpoilsOfWarList)
            print("count of cards: " + str(totalCards))
            print("\n______________________________________")
            
        return winner
    
    """
    @brief Move the first item of a deck (or list) to the back
    """
    def _MoveCardToBack(self, listToAffect):
        listToAffect.append(listToAffect.pop(0))        
        

    """
    @brief Go through the process when the player and computer drawn cards are the same
    """
    def _WarDraw(self):
        # TODO: Show three cards that were lost in the draw
        # TODO: Get the scores of the final card
        # TODO: Draw cards in the war
        # TODO: Fix issue where war occurs as the last round, and the last
        #       card is removed in the war, leaving an error when the code 
        #       returns to the _CalculateScore function
        
        print("X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X")
        print("War Check:")
        self._PrintCardCounts(self.SpoilsOfWarList)
        self._CheckCards(self.SpoilsOfWarList)
        
        
        for card in range(4):
            # Add the 4 cards currently on the table to the winning list
            if (len(self.PlayerCards) > 0):
                self.SpoilsOfWarList.append(self.PlayerCards.pop(0))
                
            if (len(self.CompCards) > 0):
                self.SpoilsOfWarList.append(self.CompCards.pop(0))
        
        totalCards = len(self.PlayerCards) + len(self.CompCards) + len(self.SpoilsOfWarList)
        print("count of cards: " + str(totalCards))
        print("End War Check")
        print("X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X")
        
        # Check the new top card to see who gets the winnings, and add the top
        # cards to the winner
        winner = self._CalculateScore(0)
        
        self._PrintCardCounts(self.SpoilsOfWarList)
        self._CheckCards(self.SpoilsOfWarList)
        
        # Add the first three cards to the deck as well
        #if (winner = "player"):
        if (winner ==  Challenger.PLAYER):
            self.PlayerCards.extend(self.SpoilsOfWarList)
            self.SpoilsOfWarList = []
            #self._PrintCardCounts()
            #self._CheckCards()
            
        #elif (winner == "comp"):
        elif (winner == Challenger.COMPUTER):
            self.CompCards.extend(self.SpoilsOfWarList)
            self.SpoilsOfWarList = []
            #self._PrintCardCounts()
            #self._CheckCards()            
            
        self._PrintCardCounts()
        self._CheckCards()
            
        return
    
    """
    @brief Second algorithm for computing outcome of a draw
    """
    def _WarDraw2(self):
        
        continueWar = True
        
        while(continueWar):
            
            # Add the top 3 cards to the spoils of war, if that many cards exist
            for card in range(3):
                if (len(self.PlayerCards) > 1):
                    self.SpoilsOfWarList.append(self.PlayerCards.pop(0))
                
                if (len(self.CompCards) > 1):
                    self.SpoilsOfWarList.append(self.CompCards.pop(0))
            
            playerCardVal = WarConstants.CARD_VALUES[self.PlayerCards[0]]
            compCardVal = WarConstants.CARD_VALUES[self.CompCards[0]]
            
            if (playerCardVal > compCardVal):
                self.SpoilsOfWarList.append(self.PlayerCards.pop(0))
                self.SpoilsOfWarList.append(self.CompCards.pop(0))
                self.PlayerCards.extend(self.SpoilsOfWarList)
                self.SpoilsOfWarList = []
                continueWar = False
                self._PrintCardCounts(self.SpoilsOfWarList)
                self._CheckCards(self.SpoilsOfWarList)
                
            elif (playerCardVal < compCardVal):
                self.SpoilsOfWarList.append(self.CompCards.pop(0))
                self.SpoilsOfWarList.append(self.PlayerCards.pop(0))
                self.CompCards.extend(self.SpoilsOfWarList)
                self.SpoilsOfWarList = []
                continueWar = False
                self._PrintCardCounts(self.SpoilsOfWarList)
                self._CheckCards(self.SpoilsOfWarList)
                
            elif (playerCardVal == compCardVal):
                self.SpoilsOfWarList.append(self.PlayerCards.pop(0))
                self.SpoilsOfWarList.append(self.CompCards.pop(0))  
                
            else:
                raise ValueError("Failure of card comparison during war")
                
        return
                
                
    
    """
    @brief Debugging function to see where the winnings of a draw go
    """
    def _GetCardsWonInWar(self, spoilsOfWar):
        winnings = ""
        for x in spoilsOfWar:
            winnings += x + ","
            
        return winnings[:len(winnings)-1]
    
    """
    @brief Redraw the graphical changes
    """
    def _DrawGraphics(self):        
        
        if (len(self.PlayerCards) > 0 and len(self.CompCards) > 0):
            # Draw new cards and display
            self.PlayerDrawnCard.setPixmap(QPixmap(WarConstants.CARD_PIC_LOC + self.PlayerCards[0]))
            self.CompDrawnCard.setPixmap(QPixmap(WarConstants.CARD_PIC_LOC + self.CompCards[0]))
        else:
            # Clear the cards, and show the winner
            #self.PlayerDrawnCard.setPixmap(None)
            #self.CompDrawnCard.setPixmap(None)
            
            #del self.PlayerCardLoc
            #del self.CompCardLoc
            
            self.WinnerLabel = QLabel(self)
            self.WinnerLabel.setText(self.Winner + " is the winner")
            self.WinnerLabel.setStyleSheet('color:blue;font-size:20;')
            self.WinnerLabel.move(self.width()/2-self.WinnerLabel.width()/2,
                                  self.height()/2 - self.WinnerLabel.height()/2)
        
        # Update the score labels
        self.PlayerScoreValLabel.setText(str(len(self.PlayerCards)))
        self.CompScoreValLabel.setText(str(len(self.CompCards)))
        
        self.show()
        
        return
