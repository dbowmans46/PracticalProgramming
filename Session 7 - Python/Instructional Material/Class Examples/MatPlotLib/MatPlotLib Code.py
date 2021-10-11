# -*- coding: utf-8 -*-
"""

Copyright 2021 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Get data in DataFrame
import pandas as pd

# Data bibliography:
#   OECD (2021), Inflation (CPI) (indicator). doi: 10.1787/eee82e6e-en (Accessed on 11 October 2021)

inflation_data_filepath = 'Inflation 1980-2020.csv'
inflation_df = pd.read_csv(inflation_data_filepath)

# Remove unneeded columns
inflation_df = inflation_df.drop(['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes'], axis=1)
inflation_df = inflation_df.rename(columns={'LOCATION':'Country', 'TIME':'Year', 'Value':'Inflation'})

# Create our individual lines of data
canada_inflation_df = inflation_df[inflation_df['Country'] == 'CAN']
japan_inflation_df = inflation_df[inflation_df['Country'] == 'JPN']
germany_inflation_df = inflation_df[inflation_df['Country'] == 'DEU']
usa_inflation_df = inflation_df[inflation_df['Country'] == 'USA']
china_inflation_df = inflation_df[inflation_df['Country'] == 'CHN']
france_inflation_df = inflation_df[inflation_df['Country'] == 'FRA']
italy_inflation_df = inflation_df[inflation_df['Country'] == 'ITA']
gb_inflation_df = inflation_df[inflation_df['Country'] == 'GBR']

# Basic scatter plot
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
# To plot, we first give the x values, then the y values, then how we want
# to customize the line display.
# We can customize the lines by giving symbols, whether or not to connect
# the symbols with a line, and the color to use.
#
# The plot() function separates lines of data by 3 sets of information:
#   the x-axis
#   the y-axis
#   the line style
# After the line data is given, any remaining keywords are used to control 
# the plot style overall
ax.plot(usa_inflation_df["Year"], usa_inflation_df["Inflation"], '+-b', 
        canada_inflation_df["Year"], canada_inflation_df["Inflation"], '--g',
        japan_inflation_df["Year"], japan_inflation_df["Inflation"], '.-m',
        germany_inflation_df["Year"], germany_inflation_df["Inflation"], '*-y',
        china_inflation_df["Year"], china_inflation_df["Inflation"], '^-r',
        france_inflation_df["Year"], france_inflation_df["Inflation"], 'o-',
        italy_inflation_df["Year"], italy_inflation_df["Inflation"], 'k',
        gb_inflation_df["Year"], gb_inflation_df["Inflation"], 'c',
        linewidth=1)

# We can set the lowest bound of the y-axis to make the graph easier to read
ax.set_ylim(bottom=-5)

# We can add labels to describe the data shown
plt.xlabel('Year')
plt.ylabel('Inflation')

# We can add annotations as desired, giving the x coordinate, y coordinate,
# text to display, and optional parameters.
ax.text(1999, -3, r"Negative inflation", horizontalalignment='center', fontsize=10)

# Finally, we must tell Python to show the figure
plt.show()