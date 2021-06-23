#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:55:57 2018

@author: doug
"""

import unittest
import sys
from GameWindow import GameWindow

class TestGameWindow(unittest.TestCase):
    
    def setUp(self):
        import time
        from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout
        from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton
        from PyQt5.QtGui import QPixmap, QIcon
        from PyQt5 import QtCore
        
        from PyQt5 import QtGui
        
        from WarConstants import WarConstants
        from Deck import Deck
        from GameManager import GameManager
        
        self.app = QApplication(sys.argv)
        
        self.gameMan = GameManager()
        self.gameWindow = GameWindow(200, 200, 800,500,"War Screen", self.gameMan)
        
        return
        
    def tearDown(self):
        self.gameWindow.close()
        del self.gameWindow
        
        sys.exit(self.app.exec_())
        
        return
    
    def test_MoveCardToBack(self):
        testList = [1,2,3,4]
        correctTestList = [2,3,4,1]
        
        self.assertEquals(self.gameWindow._MoveCardToBack(testList),correctTestList)
        
        return
        