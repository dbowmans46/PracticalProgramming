# -*- coding: utf-8 -*-

"""

Created on Wed Nov  6 09:44:35 2019

@author: db2309

"""
 

class mammal():

     def __init__(self):

          self.HasFur = True         

          return None
      
        
     def GetHasFur(self):         

          return self.__HasFur

          

     def sleep(self):

          # Decrease heartrate
          # Close eyes
          # Turn off conciousness         

          print('Sleeping')         

          return None    

          

     def eat(self):                   

          print('Eating')         

          return None
     

     

class dog(mammal):

    

     def __init__(self):

         super().__init__()

         self.HasSnout = True

          

          

     def pant(self):

          print('Panting')
         

          return None

     

     def bark(self):         

          print('Woof')
         

          return None

     

     def wimper(self):
         

          #play sound of a wimper
          print('wimper')
         

          return None