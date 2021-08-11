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
import time
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

    def setupUi(self):

        # TODO: Convert to not use QTDesigner and resource file

        self._translate = QtCore.QCoreApplication.translate

        self.MainWindow = QDialog()
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(1000, 800)
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
        #self.warTitle.setStyleSheet("background-image: url(:/Main/q02th.jpg);")
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

        self.computerBattleDeck = QtWidgets.QLabel(self.widget)
        self.computerBattleDeck.setObjectName("computerBattleDeck")

        self.gridLayout.addWidget(
            self.computerBattleDeck, 10, 3, 1, 1, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

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

        self.playerBattleDeck = QtWidgets.QLabel(self.widget)
        self.playerBattleDeck.setObjectName("playerBattleDeck")

        self.gridLayout.addWidget(
            self.playerBattleDeck, 11, 3, 1, 1, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

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

        self.computerBattleDeck.raise_()

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

    def setTheStage(self):
        self.setupUi()
        self.deckSetup()
    """
    @brief Create the starting deck
    """

    def deckSetup(self):
        WWGameManager.gameDeck.cards = WWWarConstants.CARD_FILE_NAMES * WWGameManager.deckCount
        self.WWInitDeck = WWInitialDeck(
            WWGameManager.gameDeck.cards, WWGameManager.playerDeck, WWGameManager.computerDeck)
        self.WWInitDeck.shuffleCards()
        self.WWInitDeck.deal()

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
        if len(WWGameManager.playerDeck.cards) < 5:
            WWGameManager.playerGraveyardDeck.shuffleCards()
            WWGameManager.playerDeck.cardTransferAll(
                WWGameManager.playerGraveyardDeck)
            WWGameManager.playerGraveyardDeck.cardTransferAll(
                WWGameManager.playerDeck)

        if len(WWGameManager.computerDeck.cards) < 5:
            WWGameManager.computerGraveyardDeck.shuffleCards()
            WWGameManager.computerDeck.cardTransferAll(
                WWGameManager.computerGraveyardDeck)
            WWGameManager.computerGraveyardDeck.cardTransferAll(
                WWGameManager.computerDeck)

        if (len(WWGameManager.playerDeck.cards) + len(WWGameManager.playerGraveyardDeck.cards) + len(WWGameManager.playerBattleDeck.cards) == 0):
            WWDataLogger.logger("Computer wins game")
            WWGameManager.winnerName = WWGameManager.compName
            self.wwgwIsActive = False

        if (len(WWGameManager.computerDeck.cards) + len(WWGameManager.computerGraveyardDeck.cards) + len(WWGameManager.computerBattleDeck.cards) == 0):
            WWDataLogger.logger("Player wins game")
            WWGameManager.winnerName = WWGameManager.playerName
            self.wwgwIsActive = False

        if ((len(WWGameManager.playerBattleDeck.cards)) == (WWWarConstants.DECK_SIZE / 2) and
                (len(WWGameManager.computerBattleDeck.cards)) == (WWWarConstants.DECK_SIZE / 2) and
                ((len(WWGameManager.playerBattleDeck.cards) + len(WWGameManager.computerBattleDeck.cards)) ==
                         (WWGameManager.deckCount * WWWarConstants.DECK_SIZE))
                ):
            WWDataLogger.logger("Ultra War!!!!!!")
            WWDataLogger.logger("Player is Defacto Winner")
            WWGameManager.winnerName = WWGameManager.playerName
            self.wwgwIsActive = False
        return None

    """
    @brief Primary event trigger for game logic
    """

    def dealButtonOnClick(self):
        #TODO add cardcountlabel update
        #print(WWGameManager.playerDeck.cards)
        self.cardCheck()

        WWGameManager.playerDeck.cardTransfer(WWGameManager.playerBattleDeck)
        WWGameManager.computerDeck.cardTransfer(
            WWGameManager.computerBattleDeck)

        self.cardCheck()

        if WWGameManager.winnerName == '':
            # Changes top card visually
            _translate = QtCore.QCoreApplication.translate

            prefix = "<html><head/><body><p><img src=\"./resources/production/"
            suffix = "\"/></p></body></html>"

            deckstyle_player = _translate(
            "MainWindow", prefix + WWGameManager.playerBattleDeck.cards[-1] + suffix)
            deckstyle_computer = _translate(
            "MainWindow", prefix + WWGameManager.computerBattleDeck.cards[-1] + suffix)
            
            self.playerBattleDeck.setText(deckstyle_player)
            self.computerBattleDeck.setText(deckstyle_computer)
            
            WWDataLogger.logger("Player Plays")
            WWDataLogger.logger(WWGameManager.playerBattleDeck.cards)
            WWDataLogger.logger("Computer Plays")
            WWDataLogger.logger(WWGameManager.computerBattleDeck.cards)
            self.cardValueManager = WWCardValueManager(
                WWGameManager.playerBattleDeck.cards[-1])
                
            self.cardValuePlayer = self.cardValueManager.GetCardValue()
            self.cardValueComputer = self.cardValueManager.NewCardValue(
                WWGameManager.computerBattleDeck.cards[-1])

            if self.cardValuePlayer == self.cardValueComputer:
                WWGameManager.warCount += 1

                if len(WWGameManager.playerDeck.cards) < 4:
                    for i in range(len(WWGameManager.playerDeck.cards)-1):
                        WWGameManager.playerDeck.cardTransfer(
                            WWGameManager.playerBattleDeck)
                else:
                    for i in range(3):
                        WWGameManager.playerDeck.cardTransfer(
                            WWGameManager.playerBattleDeck)

                if len(WWGameManager.computerDeck.cards) < 4:
                    for i in range(len(WWGameManager.computerDeck.cards)-1):
                        WWGameManager.computerDeck.cardTransfer(
                            WWGameManager.computerBattleDeck)
                else:
                    WWDataLogger.logger("WAR!!")
                    for i in range(3):
                        WWGameManager.computerDeck.cardTransfer(
                            WWGameManager.computerBattleDeck)
                if WWGameManager.warCount <= 40:
                    self.dealButtonOnClick()

                else:
                    self.cardCheck()

            elif self.cardValuePlayer > self.cardValueComputer:

                WWDataLogger.logger("Player wins")
                WWGameManager.playerBattleDeck.cardTransferAll(
                    WWGameManager.playerGraveyardDeck)
                WWGameManager.computerBattleDeck.cardTransferAll(
                    WWGameManager.playerGraveyardDeck)
                WWDataLogger.logger("PlayerGraveyard ")
                WWDataLogger.logger(WWGameManager.playerGraveyardDeck.cards)

            elif self.cardValuePlayer < self.cardValueComputer:

                WWDataLogger.logger("Computer wins")
                WWGameManager.playerBattleDeck.cardTransferAll(
                    WWGameManager.computerGraveyardDeck)
                WWGameManager.computerBattleDeck.cardTransferAll(
                    WWGameManager.computerGraveyardDeck)
                WWDataLogger.logger("ComputerGraveyard ")
                WWDataLogger.logger(WWGameManager.computerGraveyardDeck.cards)

            else:
                WWDataLogger.logger("---Something went wrong here---")
        return None

    """
    @brief Auto completes the game with one button click
    """

    def autoCompleteButtonOnClick(self):
        
        while WWGameManager.winnerName == '' and self.wwgwIsActive == True:
            time.sleep(1)
            self.MainWindow.show()
            self.dealButtonOnClick()

        return None

    """
    @brief Qutis the game from WWGameWindow
    """

    def quitButtonOnClick(self):
        WWDataLogger.logger("***Game was quit by user***")
        self.wwgwIsActive = False
        WWGameManager.winnerName = "No one, quitter"

    """
    @brief Converted from PYQT5 GUI
    @param MainWindow
    """

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate

        prefix = "<html><head/><body><p><img src=\"./resources/production/"
        suffix = "\"/></p></body></html>"

        MainWindow.setWindowTitle(self._translate("MainWindow", "MainWindow"))

        deckstyle_short = _translate(
            "MainWindow", prefix + WWGameManager.deckStyle + suffix)

        self.playerDeckMain.setText(deckstyle_short)

        self.playerDeckDiscard.setText(deckstyle_short)

        self.compDeckDiscard.setText(deckstyle_short)

        self.computerBattleDeck.setText(deckstyle_short)

        self.compDeckMain.setText(deckstyle_short)

        self.playerBattleDeck.setText(deckstyle_short)

        self.computerNameLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">" + WWGameManager.compName + "</span></p></body></html>"))

        self.warTitle.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:20pt; color:#ffffff;\">War!!!</span></p></body></html>"))

        self.palyerNameLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\"> " + WWGameManager.playerName + "</span></p></body></html>"))

        self.autoCompletPushButton.setText(
            self._translate("MainWindow", "Auto Complete"))

        self.designedByLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))

        self.quitPushButton.setText(self._translate("MainWindow", "Quit"))

        self.dealPushButton.setText(self._translate("MainWindow", "Deal"))

    """
    @brief Will render the card count
    """

    def cardCountUpdate(self):

        self.cardCountCompLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">0;</span></p></body></html>"))

        self.cardCountPlayerLabel.setText(self._translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">0;</span></p></body></html>"))
