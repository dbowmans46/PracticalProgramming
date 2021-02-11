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


if __name__ == "__main__":
    wwgm = WWGameManager()
    const = WWWarConstants()
    app = QApplication(sys.argv)
    # this will delete the "game_log.json" before running new game
    WWDataLogger()

    wwsw = WWSetupWindow()
    wwsw.setupUi(wwgm)
    while wwsw.wwswIsActive == True:
        QtCore.QCoreApplication.processEvents()

    wwgw = WWGameWindow()
    wwgw.setTheStage(wwgm)
    while wwgw.wwgwIsActive == True:
        QtCore.QCoreApplication.processEvents()

    wwvw = WWVictoryWindow()
    wwvw.setTheStage(wwvw, wwgm)
    while wwvw.wwvwIsActive == True:
        QtCore.QCoreApplication.processEvents()

    sys.exit()
    # sys.exit(app.exec_())


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
