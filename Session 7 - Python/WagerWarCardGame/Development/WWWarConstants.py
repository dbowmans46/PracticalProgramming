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
from PyQt5 import QtWidgets, QtCore

"""
@brief This is a static class
"""


class WWWarConstants():

    """
    Set standard size of deck
    """
    DECK_SIZE = 52

    """
    File location of cards
    """
    CARD_PICTURE_LOCATION = "./resources/production/"

    CARD_FILE_NAMES = ['c01.bmp',
                       'c02.bmp',
                       'c03.bmp',
                       'c04.bmp',
                       'c05.bmp',
                       'c06.bmp',
                       'c07.bmp',
                       'c08.bmp',
                       'c09.bmp',
                       'c10.bmp',
                       'c11.bmp',
                       'c12.bmp',
                       'c13.bmp',
                       'd01.bmp',
                       'd02.bmp',
                       'd03.bmp',
                       'd04.bmp',
                       'd05.bmp',
                       'd06.bmp',
                       'd07.bmp',
                       'd08.bmp',
                       'd09.bmp',
                       'd10.bmp',
                       'd11.bmp',
                       'd12.bmp',
                       'd13.bmp',
                       'h01.bmp',
                       'h02.bmp',
                       'h03.bmp',
                       'h04.bmp',
                       'h05.bmp',
                       'h06.bmp',
                       'h07.bmp',
                       'h08.bmp',
                       'h09.bmp',
                       'h10.bmp',
                       'h11.bmp',
                       'h12.bmp',
                       'h13.bmp',
                       's01.bmp',
                       's02.bmp',
                       's03.bmp',
                       's04.bmp',
                       's05.bmp',
                       's06.bmp',
                       's07.bmp',
                       's08.bmp',
                       's09.bmp',
                       's10.bmp',
                       's11.bmp',
                       's12.bmp',
                       's13.bmp']

    CARD_FILE_NAMES_TEST = ['A01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'E01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'c01.bmp',
                            'B01.bmp']

    """
    @brief Card style Dictionary 
    @parma key, value
    """
    deckStyle = {"BLUE": "blueBackVert.bmp",
                 "RED": "redBackVert.bmp",
                 "CAMO": "camo.bmp"}

    """
    @ brief string used on all GUI windows, A credit to the developers
    """

    """
    @ brief Retrieve users resolution to use for window size
    """

