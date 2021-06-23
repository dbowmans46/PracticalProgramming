#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:20:12 2018

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

class WarConstants():
    
    """
    @brief the size of a standard deck of cards
    """
    DECK_SIZE = 52
    
    """
    @brief The standard distance from the side borders of the window items 
           should be placed
    """
    BORDER_WIDTH_BUFFER_SIZE = 20
    
    """
    @brief The standard distance from the top and bottom borders of the window 
           items should be placed
    """
    BORDER_HEIGHT_BUFFER_SIZE = 20
    
    """
    @brief The directory where the card images are stored
    """
    CARD_PIC_LOC = "./resources/production/"
    
    """
    @brief The filenames of all the card images in a standard deck of cards
    """
    CARD_FILENAMES =   ['c01.bmp',
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
    
    CARD_VALUES = { 'c01.bmp' : 14,
                    'c02.bmp' : 2,
                    'c03.bmp' : 3,
                    'c04.bmp' : 4,
                    'c05.bmp' : 5,
                    'c06.bmp' : 6,
                    'c07.bmp' : 7,
                    'c08.bmp' : 8,
                    'c09.bmp' : 9,
                    'c10.bmp' : 10,
                    'c11.bmp' : 11,
                    'c12.bmp' : 12,
                    'c13.bmp' : 13,
                    'd01.bmp' : 14,
                    'd02.bmp' : 2,
                    'd03.bmp' : 3,
                    'd04.bmp' : 4,
                    'd05.bmp' : 5,
                    'd06.bmp' : 6,
                    'd07.bmp' : 7,
                    'd08.bmp' : 8,
                    'd09.bmp' : 9,
                    'd10.bmp' : 10,
                    'd11.bmp' : 11,
                    'd12.bmp' : 12,
                    'd13.bmp' : 13,
                    'h01.bmp' : 14,
                    'h02.bmp' : 2,
                    'h03.bmp' : 3,
                    'h04.bmp' : 4,
                    'h05.bmp' : 5,
                    'h06.bmp' : 6,
                    'h07.bmp' : 7,
                    'h08.bmp' : 8,
                    'h09.bmp' : 9,
                    'h10.bmp' : 10,
                    'h11.bmp' : 11,
                    'h12.bmp' : 12,
                    'h13.bmp' : 13,
                    's01.bmp' : 14,
                    's02.bmp' : 2,
                    's03.bmp' : 3,
                    's04.bmp' : 4,
                    's05.bmp' : 5,
                    's06.bmp' : 6,
                    's07.bmp' : 7,
                    's08.bmp' : 8,
                    's09.bmp' : 9,
                    's10.bmp' : 10,
                    's11.bmp' : 11,
                    's12.bmp' : 12,
                    's13.bmp' : 13}