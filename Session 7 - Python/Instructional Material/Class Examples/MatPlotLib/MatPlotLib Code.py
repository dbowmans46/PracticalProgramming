# -*- coding: utf-8 -*-
"""

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import matplotlib.pyplot as plt

#############################################################
#                                                           #
#           Basic Scatter Plot with Single Line             #
#                                                           #
#############################################################


# # Setup data for graph
# profit = [800, 1200, 1300, 1600, 1600, 900, 1900, 900, 1600, 1550, 2100, 2300]
# time_in_months = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# # To plot, we first give the x values, then the y values, then how we want
# # to customize the line display.
# # We can customize the lines by giving symbols, whether or not to connect
# # the symbols with a line, and the color to use.
# #
# # The plot() function separates lines of data by 3 sets of information:
# #   the x-axis
# #   the y-axis
# #   the line style
# # After the line data is given, any remaining keywords are used to control 
# # the plot style overall
# plt.plot(time_in_months,profit)  # Create plot with x,y data, but don't show yet
# plt.xlabel('Time (Months)')      # x-axis label
# plt.ylabel('Profit (1000s USD)') # y-axis label
# plt.show()                       # Show the plot

# # We can change the line style
# plt.figure(0)  # Create a new figure to show the data on
# # * is the symbol
# # the 2nd character of '-' connects the points
# # last character is color
# plt.plot(time_in_months,profit, '*-r')
# plt.xlabel('Time (Months)')
# plt.ylabel('Profit (1000s USD)')
# plt.grid(True)    # Turn x and y grid lines on
# plt.ylim(0,2500)  # Set the y-axis limits
# plt.title('Monthly Profit in 2015')
# plt.show()

# # We see some explicit marker and line styles in this example
# # markers: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
# # linestyles: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# plt.figure(1)
# plt.plot(time_in_months,profit, marker=4, linewidth=1, linestyle='-.')
# plt.xlabel('Time (Months)')
# plt.ylabel('Profit (1000s USD)')
# plt.grid(True)    # Turn x and y grid lines on
# plt.ylim(0,2500)  # Set the y-axis limits
# plt.title('Monthly Profit in 2015')
# plt.show()


#############################################################
#                                                           #
#          Scatter plot with multiple data lines            #
#                                                           #
#############################################################

# # Get data in DataFrame
# import pandas as pd

# # Data bibliography:
# #   OECD (2021), Inflation (CPI) (indicator). doi: 10.1787/eee82e6e-en (Accessed on 11 October 2021)
# inflation_data_filepath = 'Inflation 1980-2020.csv'
# inflation_df = pd.read_csv(inflation_data_filepath)

# # Remove unneeded columns
# inflation_df = inflation_df.drop(['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes'], axis=1)
# inflation_df = inflation_df.rename(columns={'LOCATION':'Country', 'TIME':'Year', 'Value':'Inflation'})

# # Create our individual lines of data
# canada_inflation_df = inflation_df[inflation_df['Country'] == 'CAN']
# japan_inflation_df = inflation_df[inflation_df['Country'] == 'JPN']
# germany_inflation_df = inflation_df[inflation_df['Country'] == 'DEU']
# usa_inflation_df = inflation_df[inflation_df['Country'] == 'USA']
# china_inflation_df = inflation_df[inflation_df['Country'] == 'CHN']
# france_inflation_df = inflation_df[inflation_df['Country'] == 'FRA']
# italy_inflation_df = inflation_df[inflation_df['Country'] == 'ITA']
# gb_inflation_df = inflation_df[inflation_df['Country'] == 'GBR']

# plt.figure(2)
# plt.plot(usa_inflation_df["Year"], usa_inflation_df["Inflation"], '+-b', 
#         canada_inflation_df["Year"], canada_inflation_df["Inflation"], '--g',
#         japan_inflation_df["Year"], japan_inflation_df["Inflation"], '.-m',
#         germany_inflation_df["Year"], germany_inflation_df["Inflation"], '*-y',
#         china_inflation_df["Year"], china_inflation_df["Inflation"], '^-r',
#         france_inflation_df["Year"], france_inflation_df["Inflation"], 'o-',
#         italy_inflation_df["Year"], italy_inflation_df["Inflation"], 'k',
#         gb_inflation_df["Year"], gb_inflation_df["Inflation"], 'c',
#         linewidth=1)

# # We can set the lowest bound of the y-axis to make the graph 
# # easier to read.  LetMatPlotLib decide the upper bounds.
# plt.ylim(bottom=-5)

# # We can add labels to describe the data shown
# plt.xlabel('Year')
# plt.ylabel('Inflation')

# # We can add annotations as desired, giving the x coordinate, y coordinate,
# # text to display, and optional parameters.
# plt.text(1999, -3, r"Negative inflation", horizontalalignment='center', fontsize=10)

# # Add a legend to depict each line
# # We can manually add the legend items, and 
# # manually dictate the location
# plt.legend(['USA','Canada','Japan','German','China','France','Italy','Great Britain'], loc='upper right')

# # Or we can use the country column from our DataFrame
# #plt.legend(inflation_df['Country'].drop_duplicates())

# # Finally, we must tell Python to show the figure
# plt.show()



#############################################################
#                                                           #
#             Multiple graphs on one figure                 #
#                                                           #
#############################################################

# plt.figure(3)
# plt.subplot(3,3,1)
# plt.plot(usa_inflation_df["Year"], usa_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('USA Inflation')

# plt.subplot(3,3,2)
# plt.plot(canada_inflation_df["Year"], canada_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('Canada Inflation')

# plt.subplot(3,3,3)
# plt.plot(japan_inflation_df["Year"], japan_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('Japan Inflation')

# plt.subplot(3,3,4)
# plt.plot(germany_inflation_df["Year"], germany_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('Germany Inflation')

# plt.subplot(3,3,5)
# plt.plot(china_inflation_df["Year"], china_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('China Inflation')

# plt.subplot(3,3,6)
# plt.plot(france_inflation_df["Year"], france_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('France Inflation')

# plt.subplot(3,3,7)
# plt.plot(italy_inflation_df["Year"], italy_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('Italy Inflation')

# plt.subplot(3,3,8)
# plt.plot(gb_inflation_df["Year"], gb_inflation_df["Inflation"])
# plt.xlabel('Year')
# plt.ylabel('Inflation')
# plt.title('Great Britain Inflation')

# # Setup the spacing so the graphs aren't on top of each other
# # top    - The position of the top edge of the subplots, as a fraction of the figure height.
# # bottom - The position of the right edge of the subplots, as a fraction of the figure width.
# # left   - The position of the left edge of the subplots, as a fraction of the figure width.
# # right  - The position of the right edge of the subplots, as a fraction of the figure width.
# # hspace - The height of the padding between subplots, as a fraction of the average Axes height.
# # wspace - The width of the padding between subplots, as a fraction of the average Axes width.
# # plt.subplots_adjust(top=1, bottom=0.1, left=0.10, right=0.95, hspace=1.25, wspace=0.75)

# plt.show()



#############################################################
#                                                           #
#              Exporting figures to a file                  #
#                                                           #
#############################################################

# plt.figure(4)
# profit = [800, 1200, 1300, 1600, 1600, 900, 1900, 900, 1600, 1550, 2100, 2300]
# time_in_months = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# plt.plot(time_in_months,profit, '*-r')
# plt.xlabel('Time (Months)')
# plt.ylabel('Profit (1000s USD)')
# plt.grid(True)    # Turn x and y grid lines on
# plt.ylim(0,2500)  # Set the y-axis limits
# plt.title('Monthly Profit in 2015')

# # Save the file as a PDF
# plt.savefig('2015_profit.pdf')

# # Save the file as a PNG
# plt.savefig('2015_profit.png')




#############################################################
#                                                           #
#            Examples of Different Plot Types               #
#                                                           #
#############################################################

# More examples at https://matplotlib.org/stable/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py

fig = plt.figure(5)
# Scatter plot
plt.subplot(3,3,1)
plt.plot([1,2,3,4,5],[1,4,9,16,25],'.b')
plt.xlabel('x')
plt.ylabel('$x^2$', rotation=0) # Rotates the y-label
plt.xticks(fontsize=5)
plt.yticks(fontsize=6)
plt.title('Scatter Plot')

# Histogram
plt.subplot(3,3,2)
data = [1,5,2,3,3,5,8,5,5,6,5,0,4,4,7,3,9,7,6,8,9]
num_bins = 10
plt.hist(data, num_bins, density=True)
plt.xlabel('#')
plt.ylabel('Frequency')
plt.xticks(ticks=range(10), fontsize=5)
plt.yticks(fontsize=5)
plt.title('Histogram')

# 3D Surface
from matplotlib import cm
import numpy as np
from mpl_toolkits import mplot3d
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.cos(R)

# Plot the surface
ax = fig.add_subplot(3,3,3,projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(fontsize=5)
plt.yticks(fontsize=6)
plt.title('3-D Surface')

# Pie chart
plt.subplot(3,3,4)
plt.pie([1,2,3,4,5])
plt.title('Pie Chart')
          
# Vertical Bar
plt.subplot(3,3,5)
profit = [800, 1200, 1300, 1600, 1600, 900, 1900, 900, 1600, 1550, 2100, 2300]
months = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.bar(months,profit)
plt.xticks(fontsize=5, rotation=90) # Rotates the tick labels so they don't overlap each other
plt.yticks(fontsize=6)
plt.xlabel('Month')
plt.ylabel('Profit')
plt.title('Vertical Bar Chart')

# Horizontal Bar
plt.subplot(3,3,6)
profit = [800, 1200, 1300, 1600, 1600, 900, 1900, 900, 1600, 1550, 2100, 2300]
months = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.barh(months,profit)
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=6)
plt.xlabel('Month')
plt.ylabel('Profit')
plt.title('Vertical Bar Chart')

# Polar
from math import pi
plt.subplot(3,3,7, projection='polar')
radius = [0,1,2,3,4,5]
theta = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]
plt.polar(theta, radius)
plt.xticks(fontsize=5)
plt.yticks(range(6),fontsize=5)
plt.title('Polar Plot')

# Adjust the spacing between subplots
plt.subplots_adjust(top=1.2, hspace=1.25, wspace=0.75)
plt.show()
