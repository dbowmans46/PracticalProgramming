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

# TODO Need to get the "Play again button working"

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
@brief Creates victory window.
"""


class WWVictoryWindow(object):

    def setupUi(self, wwVictoryWindow, wwGameManager):
        self._translate = QtCore.QCoreApplication.translate

        self.wwvw = wwVictoryWindow
        self.wwgm = wwGameManager
        self.MainWindow = QDialog()

        self.MainWindow.setObjectName("WWVictoryWindow")
        self.MainWindow.resize(800, 600)
        self.MainWindow.setStyleSheet("background-color: rgb(0, 85, 0);")

        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 20, 800, 600))
        self.widget.setObjectName("widget")

        self.victorLabel = QtWidgets.QLabel(self.widget)
        self.victorLabel.setGeometry(QtCore.QRect(270, 10, 131, 51))
        self.victorLabel.setObjectName("victorLabel")

        self.trophyPixMap = QtWidgets.QLabel(self.widget)
        self.trophyPixMap.setGeometry(QtCore.QRect(200, 70, 290, 340))
        self.trophyPixMap.setStyleSheet(
            "border-image: url(:/Main/img/trophy.png); background-color: rgba(255, 255, 255, 0);")
        self.trophyPixMap.setObjectName("trophyPixMap")

        self.playAgainPushButton = QtWidgets.QPushButton(self.widget)
        self.playAgainPushButton.setGeometry(QtCore.QRect(210, 480, 75, 23))
        self.playAgainPushButton.setStyleSheet(
            "color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.playAgainPushButton.setObjectName("playAgainPushButton")
        self.playAgainPushButton.clicked.connect(self.playAgainButtonOnClick)

        self.quitPushButton = QtWidgets.QPushButton(self.widget)
        self.quitPushButton.setGeometry(QtCore.QRect(410, 480, 75, 23))
        self.quitPushButton.setStyleSheet(
            "color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.quitPushButton.setObjectName("quitPushButton")
        self.quitPushButton.clicked.connect(self.quitButtonOnClick)

        self.designedByLabel = QtWidgets.QLabel(self.widget)
        self.designedByLabel.setGeometry(QtCore.QRect(250, 500, 161, 41))
        self.designedByLabel.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0);")
        self.designedByLabel.setObjectName("designedByLabel")

        self.winnerLabel = QtWidgets.QLabel(self.widget)
        self.winnerLabel.setGeometry(QtCore.QRect(260, 420, 281, 31))
        self.winnerLabel.setAutoFillBackground(False)
        self.winnerLabel.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0);")
        self.winnerLabel.setObjectName("winnerLabel")

        self.victorLabel.raise_()

        self.trophyPixMap.raise_()

        self.quitPushButton.raise_()

        self.playAgainPushButton.raise_()

        self.designedByLabel.raise_()

        self.winnerLabel.raise_()

        self.retranslateUi(self.MainWindow)
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

        self.MainWindow.show()

        self.wwvwIsActive = True

    def setTheStage(self, windowItem, wwGameManager):
        self.setupUi(windowItem, wwGameManager)

    def playAgainButtonOnClick(self):
        self.wwgm.playAgainToggle = True
        self.wwvwIsActive = False 
        # self.wwgm = WWGameManager()
        # sys.exit(self)  # exit current session
        # WWOverlord()  # restart at the beginning
        # print("test")

    def quitButtonOnClick(self):
        WWDataLogger.logger("***Game was quit by user***")
        self.wwvwIsActive = False
        self.MainWindow.close()

    def retranslateUi(self, WWVictoryWindow):
        _translate = QtCore.QCoreApplication.translate

        WWVictoryWindow.setWindowTitle(_translate("WWVictoryWindow", "Dialog"))

        self.victorLabel.setText(_translate(
            "WWVictoryWindow", "<html><head/><body><p><span style=\" font-size:24pt; color:#ffffff;\">Victory!!!</span></p></body></html>"))

        self.trophyPixMap.setText(_translate(
            "WWVictoryWindow", "<html><head/><body><p><br/></p></body></html>"))

        self.playAgainPushButton.setText(
            _translate("WWVictoryWindow", "Play Again"))

        self.quitPushButton.setText(_translate("WWVictoryWindow", "Quit"))

        self.designedByLabel.setText(_translate(
            "WWMainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))

        self.winnerLabel.setText(_translate(
            "WWMainWindow", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">"
            + self.wwgm.winnerName + " Wins!!</span></p></body></html>"))
