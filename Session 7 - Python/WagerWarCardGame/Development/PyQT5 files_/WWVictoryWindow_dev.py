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
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WWVictoryWindow(object):
    def setupUi(self, WWVictoryWindow):
        
        WWVictoryWindow.setObjectName("WWVictoryWindow")
        WWVictoryWindow.resize(800, 600)
        WWVictoryWindow.setStyleSheet("background-color: rgb(0, 85, 0);")
        
        self.victorLabel = QtWidgets.QLabel(WWVictoryWindow)
        self.victorLabel.setGeometry(QtCore.QRect(300, 10, 131, 51))
        self.victorLabel.setObjectName("victorLabel")
        
        self.trophyPixMap = QtWidgets.QLabel(WWVictoryWindow)
        self.trophyPixMap.setGeometry(QtCore.QRect(190, 70, 291, 341))
        self.trophyPixMap.setStyleSheet("border-image: url(:/Main/img/trophy.png); background-color: rgba(255, 255, 255, 0);")
        self.trophyPixMap.setObjectName("trophyPixMap")
        
        self.playAgainPushButton = QtWidgets.QPushButton(WWVictoryWindow)
        self.playAgainPushButton.setGeometry(QtCore.QRect(220, 480, 75, 23))
        self.playAgainPushButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.playAgainPushButton.setObjectName("playAgainPushButton")
        
        self.quitPushButton = QtWidgets.QPushButton(WWVictoryWindow)
        self.quitPushButton.setGeometry(QtCore.QRect(410, 480, 75, 23))
        self.quitPushButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.quitPushButton.setObjectName("quitPushButton")
        
        self.designedByLabel = QtWidgets.QLabel(WWVictoryWindow)
        self.designedByLabel.setGeometry(QtCore.QRect(270, 550, 161, 41))
        self.designedByLabel.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.designedByLabel.setObjectName("designedByLabel")
        
        self.winnerLabel = QtWidgets.QLabel(WWVictoryWindow)
        self.winnerLabel.setGeometry(QtCore.QRect(220, 420, 281, 31))
        self.winnerLabel.setAutoFillBackground(False)
        self.winnerLabel.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.winnerLabel.setObjectName("winnerLabel")
        
        self.victorLabel.raise_()
        
        self.trophyPixMap.raise_()
        
        self.quitPushButton.raise_()
        
        self.playAgainPushButton.raise_()
        
        self.designedByLabel.raise_()
        
        self.winnerLabel.raise_()

        self.retranslateUi(WWVictoryWindow)
        QtCore.QMetaObject.connectSlotsByName(WWVictoryWindow)

    def retranslateUi(self, WWVictoryWindow):
        _translate = QtCore.QCoreApplication.translate
        WWVictoryWindow.setWindowTitle(_translate("WWVictoryWindow", "Dialog"))
        self.victorLabel.setText(_translate("WWVictoryWindow", "<html><head/><body><p><span style=\" font-size:24pt; color:#ffffff;\">Victory!!!</span></p></body></html>"))
        self.trophyPixMap.setText(_translate("WWVictoryWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.playAgainPushButton.setText(_translate("WWVictoryWindow", "Play Again"))
        self.quitPushButton.setText(_translate("WWVictoryWindow", "Quit"))
        self.designedByLabel.setText(_translate("WWVictoryWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))
        self.winnerLabel.setText(_translate("WWVictoryWindow", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">&lt;Winner Name&gt; Wins!!</span></p></body></html>"))
