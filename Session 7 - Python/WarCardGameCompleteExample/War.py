# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:00:33 2018

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

import sys

# Imports needed to get the basic window
from PyQt5.QtWidgets import QApplication, QWidget

from WarConstants import WarConstants
from SetupWindow import SetupWindow
from GameManager import GameManager
from GameWindow import GameWindow

# Setup test classes

"""
@brief Method to test that the GameManager was set correctly
@param gameManager The GameManager to test
"""
def _TestGameManagerSetup(gameManager):
    print(gameManager.DeckCount)
    print(gameManager.PlayerName)
    print(gameManager.CompName)


# TODO: Create and import a class to manage the game logic

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    
    # Setup the game manager class
    
    gameMan = GameManager()
    
    # Create the start screen to setup the game constants
    print("Starting setup window")
    setupWindow = SetupWindow(gameMan, 400,300,200,400, "Setup the Battle")
    
    gameMan.PlayerName = setupWindow.PlayerName
    gameMan.CompName = setupWindow.CompName
    gameMan.DeckCount = int(setupWindow.DeckCount)
    
    # Close the setup window
    setupWindow.close()
    print("Setup window closed")
    
    #_TestGameManagerSetup(gameMan)
    #print("Game code has continued")
    
    # Setup the gameplay window
    print("Starting main game window")
    gameWindow = GameWindow(200, 200, 800,500,"War Screen", gameMan)
    gameWindow.close()
    print("Closed main game window")
    sys.exit(app.exec_())
    print("App exited")
    