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
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QLineEdit, QPushButton, QComboBox, QDialog
from WWCardValueManager import WWCardValueManager
from WWWarConstants import WWWarConstants
from WWInitialDeck import WWInitialDeck
from WWGameManager import WWGameManager
from WWCardsDeck import WWCardsDeck
from WWDataLogger import WWDataLogger

"""
@brief Creates game window, handles game logic
"""


class WWGameWindow(object):
    """
    @brief Using PYQT5, creates GUI for game window, handles connnections between buttons
    @param wwGameManager
    """

    def setupUi(self, wwGameManager):
        self._translate = QtCore.QCoreApplication.translate

        self.wwgm = wwGameManager
        self.MainWindow = QDialog()

        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(800, 600)
        self.MainWindow.setStyleSheet("background-color: rgb(0, 85, 0);")

        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 20, 711, 531))
        self.widget.setObjectName("widget")

        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.playerDeckMain = QtWidgets.QLabel(self.widget)
        self.playerDeckMain.setObjectName("playerDeckMain")

        self.gridLayout.addWidget(
            self.playerDeckMain, 11, 2, 1, 1, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.computerNameLabel = QtWidgets.QLabel(self.widget)
        self.computerNameLabel.setObjectName("computerNameLabel")

        self.gridLayout.addWidget(self.computerNameLabel, 4, 3, 1, 1)

        self.warTitle = QtWidgets.QLabel(self.widget)
        self.warTitle.setStyleSheet("background-image: url(:/Main/q02th.jpg);")
        self.warTitle.setObjectName("warTitle")

        self.gridLayout.addWidget(self.warTitle, 2, 3, 1, 1)

        self.playerDeckDiscard = QtWidgets.QLabel(self.widget)
        self.playerDeckDiscard.setObjectName("playerDeckDiscard")

        self.gridLayout.addWidget(
            self.playerDeckDiscard, 11, 4, 1, 1, QtCore.Qt.AlignHCenter)

        self.compDeckDiscard = QtWidgets.QLabel(self.widget)
        self.compDeckDiscard.setObjectName("compDeckDiscard")

        self.gridLayout.addWidget(
            self.compDeckDiscard, 10, 4, 1, 1, QtCore.Qt.AlignHCenter)

        self.compDeckActive = QtWidgets.QLabel(self.widget)
        self.compDeckActive.setObjectName("compDeckActive")

        self.gridLayout.addWidget(
            self.compDeckActive, 10, 3, 1, 1, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.palyerNameLabel = QtWidgets.QLabel(self.widget)
        self.palyerNameLabel.setObjectName("palyerNameLabel")

        self.verticalLayout.addWidget(self.palyerNameLabel)

        self.gridLayout.addLayout(self.verticalLayout, 14, 3, 1, 1)

        self.compDeckMain = QtWidgets.QLabel(self.widget)
        self.compDeckMain.setObjectName("compDeckMain")

        self.gridLayout.addWidget(
            self.compDeckMain, 10, 2, 1, 1, QtCore.Qt.AlignHCenter)

        self.autoCompletPushButton = QtWidgets.QPushButton(self.widget)
        self.autoCompletPushButton.setStyleSheet(
            "color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.autoCompletPushButton.setObjectName("autoCompletPushButton")
        self.autoCompletPushButton.clicked.connect(
            self.autoCompleteButtonOnClick)

        self.gridLayout.addWidget(self.autoCompletPushButton, 16, 2, 1, 1)

        self.playerDeckActive = QtWidgets.QLabel(self.widget)
        self.playerDeckActive.setObjectName("playerDeckActive")

        self.gridLayout.addWidget(
            self.playerDeckActive, 11, 3, 1, 1, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.designedByLabel = QtWidgets.QLabel(self.widget)
        self.designedByLabel.setObjectName("DesignedByLabel")

        self.gridLayout.addWidget(
            self.designedByLabel, 18, 1, 1, 6, QtCore.Qt.AlignHCenter)

        self.quitPushButton = QtWidgets.QPushButton(self.widget)
        self.quitPushButton.setStyleSheet(
            "color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.quitPushButton.setObjectName("quitPushButton")
        self.quitPushButton.clicked.connect(self.quitButtonOnClick)

        self.gridLayout.addWidget(
            self.quitPushButton, 16, 4, 1, 1, QtCore.Qt.AlignHCenter)

        self.dealPushButton = QtWidgets.QPushButton(self.widget)
        self.dealPushButton.setStyleSheet(
            "color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.dealPushButton.setObjectName("dealPushButton")
        self.dealPushButton.clicked.connect(self.dealButtonOnClick)

        self.gridLayout.addWidget(
            self.dealPushButton, 16, 3, 1, 1, QtCore.Qt.AlignHCenter)

        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.gridLayout.addLayout(self.verticalLayout_3, 10, 1, 1, 1)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.gridLayout.addLayout(self.verticalLayout_2, 2, 1, 1, 1)

        self.cardCountCompLabel = QtWidgets.QLabel(self.widget)
        self.cardCountCompLabel.setObjectName("cardCountCompLabel")

        self.gridLayout.addWidget(self.cardCountCompLabel, 5, 3, 1, 1)

        self.cardCountPlayerLabel = QtWidgets.QLabel(self.widget)
        self.cardCountPlayerLabel.setObjectName("cardCountPlayerLabel")

        self.gridLayout.addWidget(self.cardCountPlayerLabel, 15, 3, 1, 1)

        self.autoCompletPushButton.raise_()

        self.dealPushButton.raise_()

        self.quitPushButton.raise_()

        self.warTitle.raise_()

        self.playerDeckDiscard.raise_()

        self.designedByLabel.raise_()

        self.compDeckActive.raise_()

        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")

        self.retranslateUi(self.MainWindow)
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

        self.MainWindow.show()

        self.dealPushButton.clicked.connect(self.turnEventUpdate)

        self.quitPushButton.clicked.connect(self.turnEventUpdate)

        self.autoCompletPushButton.clicked.connect(self.turnEventUpdate)

        self.wwgwIsActive = True

    '''
    @brief sets up the game state by shuffling deck and dealing cards to player and computer. 
    @param
    '''

    def setTheStage(self, wwgm):
        self.setupUi(wwgm)
        self.deckSetup()

    # Pass list of cards * # of decks to initialDeck, collection of all cards used in game.
    """
    @brief Create the starting deck
    """

    def deckSetup(self):
        # multiply start deck by number of decks selected by player.
        self.wwgm.gameDeck.cards = WWWarConstants.CARD_FILE_NAMES_TEST * self.wwgm.deckCount
        self.WWInitDeck = WWInitialDeck(
            self.wwgm.gameDeck.cards, self.wwgm.playerDeck, self.wwgm.computerDeck)
       # [WWDataLogger.logger(x) for x in self.WWInitDeck.cards]
        self.WWInitDeck.shuffleCards()

        self.WWInitDeck.deal()
        #WWDataLogger.logger("Player Deck Size:" + str(len(self.wwgm.playerDeck.cards)))
        #WWDataLogger.logger("Computer Deck Size:", len(self.wwgm.computerDeck.cards))

    """
    @brief Update GUI: update Player/Comp Card Count, Turn Count, Populate the new card image.
    @param
    """

    def turnEventUpdate(self):

        return None
    """
    @brief Checks count of player and computer to determine when the graveyards need to be shuffled in.
    """

    def cardCheck(self):        
        if len(self.wwgm.playerDeck.cards) < 5:
            self.wwgm.playerGraveyardDeck.shuffleCards()
            self.wwgm.playerDeck.cardTransferAll(self.wwgm.playerGraveyardDeck)
            self.wwgm.playerGraveyardDeck.cardTransferAll(self.wwgm.playerDeck)

        if len(self.wwgm.computerDeck.cards) < 5:
            self.wwgm.computerGraveyardDeck.shuffleCards()
            self.wwgm.computerDeck.cardTransferAll(
                self.wwgm.computerGraveyardDeck)
            self.wwgm.computerGraveyardDeck.cardTransferAll(
                self.wwgm.computerDeck)

        if (len(self.wwgm.playerDeck.cards) + len(self.wwgm.playerGraveyardDeck.cards) + len(self.wwgm.playerBattleDeck.cards) == 0):
            # declare computer the winner
            # go to victory window
            WWDataLogger.logger("Computer wins game")
            self.wwgm.winnerName = self.wwgm.compName
            self.wwgwIsActive = False
            self.MainWindow.close()

        if (len(self.wwgm.computerDeck.cards) + len(self.wwgm.computerGraveyardDeck.cards) + len(self.wwgm.computerBattleDeck.cards) == 0):
            # declare player the winner
            # go to victory window
            WWDataLogger.logger("Player wins game")
            self.wwgm.winnerName = self.wwgm.playerName
            self.wwgwIsActive = False
            self.MainWindow.close()

        # Only occurs when the last hand is a war and ALL cards are in the battle decks.
        #TODO compare each deck with DECK.SIZe / 2 
        if ((len(self.wwgm.playerBattleDeck.cards) + (len(self.wwgm.computerBattleDeck.cards))) == self.wwgm.deckCount * WWWarConstants.DECK_SIZE):
            WWDataLogger.logger("Ultra War!!!!!!")
            WWDataLogger.logger("Player is Defacto Winner")
            self.wwgm.winnerName = self.wwgm.playerName
            self.wwgwIsActive = False
            self.MainWindow.close()
        return None

    """
    @brief Primary event trigger for game logic
    """

    def dealButtonOnClick(self):

        self.cardCheck()

        # check if a winner name has been determined, if so, end loop.
        # if self.wwgm.winnerName != '':
        #     # sys.stdout.close()  # closes text file
        #     return None

        # Transfer top cards from Player/Computer Library to battlefield

        self.wwgm.playerDeck.cardTransfer(self.wwgm.playerBattleDeck)
        self.wwgm.computerDeck.cardTransfer(self.wwgm.computerBattleDeck)
        self.cardCheck()
        if self.wwgm.winnerName == '':
            # func to log info to game_log.txt
            WWDataLogger.logger("Player Plays")
            WWDataLogger.logger(self.wwgm.playerBattleDeck.cards)
            WWDataLogger.logger("Computer Plays")
            WWDataLogger.logger(self.wwgm.computerBattleDeck.cards)
            print("Printing. Deal Button on Click")
            self.cardValueManager = WWCardValueManager(
                self.wwgm.playerBattleDeck.cards[-1])
            self.cardValuePlayer = self.cardValueManager.GetCardValue()
            self.cardValueComputer = self.cardValueManager.NewCardValue(
                self.wwgm.computerBattleDeck.cards[-1])
    
            # Compare computerBattle and playerBattle
            if self.cardValuePlayer == self.cardValueComputer:
                self.wwgm.warCount += 1
    
                # Check for less than three cards if less throw in all but one card.
                if len(self.wwgm.playerDeck.cards) < 4:
                    for i in range(len(self.wwgm.playerDeck.cards)-1):
                        self.wwgm.playerDeck.cardTransfer(
                            self.wwgm.playerBattleDeck)
                else:
                    for i in range(3):
                        self.wwgm.playerDeck.cardTransfer(
                            self.wwgm.playerBattleDeck)
    
                if len(self.wwgm.computerDeck.cards) < 4:
                    for i in range(len(self.wwgm.computerDeck.cards)-1):
                        self.wwgm.computerDeck.cardTransfer(
                            self.wwgm.computerBattleDeck)
                else:
                    WWDataLogger.logger("WAR!!")
                    for i in range(3):
                        self.wwgm.computerDeck.cardTransfer(
                            self.wwgm.computerBattleDeck)
                if self.wwgm.warCount <= 40:
                    print("The war count is:", self.wwgm.warCount)
                    self.dealButtonOnClick()
    
                else:
                    self.wwgm.winnerName = "Ultimate Draw"
                    self.wwgwIsActive = False
                    self.MainWindow.close()
    
            elif self.cardValuePlayer > self.cardValueComputer:
    
                WWDataLogger.logger("Player wins")
                self.wwgm.playerBattleDeck.cardTransferAll(
                    self.wwgm.playerGraveyardDeck)
                self.wwgm.computerBattleDeck.cardTransferAll(
                    self.wwgm.playerGraveyardDeck)
                WWDataLogger.logger("PlayerGraveyard ")
                WWDataLogger.logger(self.wwgm.playerGraveyardDeck.cards)
    
            elif self.cardValuePlayer < self.cardValueComputer:
    
                WWDataLogger.logger("Computer wins")
                self.wwgm.playerBattleDeck.cardTransferAll(
                    self.wwgm.computerGraveyardDeck)
                self.wwgm.computerBattleDeck.cardTransferAll(
                    self.wwgm.computerGraveyardDeck)
                WWDataLogger.logger("ComputerGraveyard ")
                WWDataLogger.logger(self.wwgm.computerGraveyardDeck.cards)
    
            else:
                WWDataLogger.logger("---Something went wrong here---")
        return None

    """
    @brief Auto completes the game with one button click
    """

    def autoCompleteButtonOnClick(self):
        while self.wwgm.winnerName == '' and self.wwgwIsActive == True:
            self.dealButtonOnClick()
            print("autoCompleteButtonOnClick called - while loop")
        '''
        else:
            WWDataLogger.logger("Loop error in autoComplete")
        '''
        return None

    """
    @brief Qutis the game from WWGameWindow
    """

    def quitButtonOnClick(self):
        WWDataLogger.logger("***Game was quit by user***")
        sys.exit(self)

    """
    @brief Converted from PYQT5 GUI
    @param MainWindow
    """

    def retranslateUi(self, MainWindow):

        MainWindow.setWindowTitle(self._translate("MainWindow", "MainWindow"))

        self.playerDeckMain.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.computerNameLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">" + self.wwgm.compName + "</span></p></body></html>"))

        self.warTitle.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:20pt; color:#ffffff;\">War!!!</span></p></body></html>"))

        self.playerDeckDiscard.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.compDeckDiscard.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.compDeckActive.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.palyerNameLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\"> " + self.wwgm.playerName + "</span></p></body></html>"))

        self.compDeckMain.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.autoCompletPushButton.setText(
            self._translate("MainWindow", "Auto Complete"))

        self.playerDeckActive.setText(self._translate(
            "MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))

        self.designedByLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))

        self.quitPushButton.setText(self._translate("MainWindow", "Quit"))

        self.dealPushButton.setText(self._translate("MainWindow", "Deal"))

    """
    @brief Will render the card count?
    """

    def cardCountUpdate(self):

        self.cardCountCompLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">Card Count: + &lt;int&gt;</span></p></body></html>"))

        self.cardCountPlayerLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">Card Count: + &lt;int&gt;</span></p></body></html>"))
