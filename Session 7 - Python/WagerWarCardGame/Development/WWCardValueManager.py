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

import re


'''
@brief calculates score/value of card based on card file name
''' 
class WWCardValueManager:
    
    '''
    @brief Constructor sets defualt value.
    @param string CardName Filename of card given (i.e c02.bmp)
    '''
    def __init__(self, CardName):
        self.CardName = CardName
        self.SetCardValue() 
 
    '''        
    @brief Using RegEx, split off the numarical string and convert to INT.
    '''    
    def SetCardValue(self):
        custom_card = "Please check custom card values against stock values (i.e C02.bmp)"
        message = ''
                                                                            
        try:
            self.CardValue = int(re.findall('[0-1]{1}[0-9]', self.CardName)[0])
            
            if self.CardValue > 14 or self.CardValue < 1: 
                raise ValueError
           
            if len(self.CardName) != 7:
                message = 'The card file name length does not equal 7'
                raise IndexError
                
        except IndexError:
            raise IndexError ("No values found or incorrect file format. " + custom_card + message) 
            
        except ValueError:            
            if self.CardValue > 14: 
                raise ValueError("Card value too high. " + custom_card)
            if self.CardValue < 1:
                raise ValueError("Card value is too Low. " + custom_card)
        except:
            raise ("An Error Occured.")
            
        return None
     
    '''
    @brief How to run independent:  set var to (WWCardValueManager), call var.method to run.
    '''    
    def GetCardValue(self):
        return self.CardValue
   
    '''
    @brief Returns new card value, after an initial run. Old card value rewritten.
    @param string CardName Filename of card given (i.e c02.bmp)
    '''
    def NewCardValue(self, CardName):
         self.CardName = CardName
         self.SetCardValue()
         return self.GetCardValue()
         
         

          

     