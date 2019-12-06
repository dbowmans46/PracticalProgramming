"""
MIT License

Copyright (c) 2019 Doug Bowman dbowmans46@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


## Functionality if condition is met
#condition = True
#if (condition):
#    print('condition is True')
#print('Exited conditional statement')
#
#
## Functionality if condition is not met
#condition2 = False
#if (condition2):
#    print('condition2 is True')
#print('Exited conditional statement')


## Check if an item belongs to a list
#pn=133148
#pn_list=['55238','133148','44345','11098','M18038-101']
#
#
#if (pn in pn_list):
#    print('Match: ' + str(pn))
#
##if-else allows you to specify an alternative, catch-all
#qty = 4
#if (qty > 4):
#    print('condition is True')
#else:
#    print('condition is False')
#
# You can nest if-else to catch more items
grade='B'
if (grade == 'A'):
    print(str(grade) + ': Outstanding')
else:
    if (grade == 'B'):
        print(str(grade) + ': Good Job')
    #Use elif to avoid unnecessary formatting (tabs)
    #and exessive syntax
    elif (grade == 'C'):
        print(str(grade) + ': Average')
    elif (grade == 'D'):
        print(str(grade) + ': You need to stop drinking so much')
    else:
        print(str(grade) + ': Your mother and I are thinking ' +
              'about sending you to a military academy...' +
              'doesn\'t that sound like fun?')
    
    
year = '2018'

if ( (year > '2000') or (year > '2020') ):
    print(year)



