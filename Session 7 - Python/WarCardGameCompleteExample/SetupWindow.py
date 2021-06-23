#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:40:07 2018

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


import sys, time
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5 import QtGui, QtCore

class SetupWindow(QWidget):
    
    '''
    @brief Constructor, setup default values
    @param left Distance of left side of window from left side of monitor screen
    @param top Distance of top side of window from top of monitor screen
    @param width The width of the window in pixels
    @param height The height of the window in pixels
    @param windowTitle The title of the setup window
    @param playerName The name of the player
    '''
    def __init__(self, gameManager, left, top, width, height, windowTitle, defaultCompName = 'Torkon the Annihilator of Souls'):
        super().__init__()
        self.Left = left
        self.Top = top
        self.Width = width
        self.Height = height
        
        self.setGeometry(self.Left, self.Top, self.Width, self.Height)
        self.setWindowTitle(windowTitle)
        
        # Setup a pause variable to hold code execution until variables are set
        self.PauseForButtonClick = True
        
        self.SetupInterface(gameManager)        
        
        
    '''
    @brief Setup default items
    @param gameManager The game manager housing all the states and behind-the-scenes data
    '''
    def SetupInterface(self, gameManager):
        self.playerNameLabel = QLabel()
        self.playerTextEdit = QLineEdit(self)
        
        self.compNameLabel = QLabel()
        self.compTextEdit = QLineEdit(self)
        
        self.deckCountLabel = QLabel()
        self.deckCount = QLineEdit(self)
        
        # Add the default names and their labels
        self.playerNameLabel.setText("Declare Your Name, Warrior:")
        self.playerTextEdit.setText(gameManager.PlayerName)
        self.compNameLabel.setText("Choose Your Challenger, Warrior:")
        self.compTextEdit.setText(gameManager.CompName)
        
        # Add holder for the count of cards
        self.deckCountLabel.setText("Select the Number of Battles in \nthe War (# of decks):")
        self.deckCount.setText(str(gameManager.DeckCount))
        
        # Add button to submit values
        self.submitButton = QPushButton()
        self.submitButton.setText("Onwards to glory!")
        self.submitButton.clicked.connect(self._SubmitButton_Clicked)
        
        # Organize elements in the vertical layout box
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.playerNameLabel)
        self.vbox.addWidget(self.playerTextEdit)
        
        # Add spacer
        self.vbox.addStretch(1)
        
        self.vbox.addWidget(self.compNameLabel)
        self.vbox.addWidget(self.compTextEdit)
        
        self.vbox.addStretch(1)
        
        self.vbox.addWidget(self.deckCountLabel)
        self.vbox.addWidget(self.deckCount)
        
        self.vbox.addStretch(1)
        
        self.vbox.addWidget(self.submitButton)        
        
        self.setLayout(self.vbox)
        
        self.show()
        
        # Wait for button to be pressed before code continues
        while (self.PauseForButtonClick == True):
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.05)
            
        
    '''
    @brief Setup events from the button
    @param gameManager The game manager housing all the states and behind-the-scenes data
    '''
    def _SubmitButton_Clicked(self):
        
        self.PlayerName = self.playerTextEdit.text()
        self.CompName = self.compTextEdit.text()
        self.DeckCount = int(self.deckCount.text())
        
        # Test the data for correct form
        
        # Continue execution of code
        self.PauseForButtonClick = False
        
    """
    @brief Method to test that the GameManager was set correctly
    @param gameManager The GameManager to test
    """
    def _TestGameManagerSetup(self, gameManager):
        print(gameManager.DeckCount)
        print(gameManager.PlayerName)
        print(gameManager.CompName)
        
        return None
        
        
    
