"""
MIT License

Copyright (c) 2019 dbowmans46

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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




from random import shuffle
from WWCardsDeck import WWCardsDeck

'''
@ brief a deck class that adds the method of "shuffling"
'''
class WWShuffleDeck(WWCardsDeck):
    
    """
    @ brief Constructor setup
    @ param string-list cards strings are filenames of card images
    """
    def __init__(self,cards):

        super().__init__(cards)
        return None
    
    """
    @ brief randomly shuffles the cards in the WWShuffleDeck object. 
    """
    def shuffleCards(self):
        # Must be a list to use shuffle
        shuffle(self.cards)
        
        return None