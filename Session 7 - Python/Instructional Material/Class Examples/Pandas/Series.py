# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Different ways to setup a series
print("Setting up a Series")
print("------------------------")
# One option
pn_series1 = pd.Series(['121115', '732002', '055097','232339'])
pn_series1.name = 'Part Numbers'

# Another option
pn_series2 = pd.Series(['121115', '732002', '055097','232339'], name="Part Numbers")

# Another Option
series_data = ['121115', '732002', '055097','232339']
pn_series3 = pd.Series(data=series_data, name="Part Numbers")

# Using an index
index_vals = ['a','b','c','d']
pn_series_with_index = pd.Series(data=series_data, name="Part Numbers", index=index_vals)


# Ways to access data
# 1: Like a list
print("Accessing series data like a list: ", pn_series3[2])
print("Accessing series with index data like a list: ", pn_series_with_index[2])

# 2: By index location (the preferred method for DataFrames)
print("Accessing series data with iloc: ", pn_series3.iloc[2])
print("Accessing series with index data like a list: ", pn_series_with_index[2])

# 3: By using the index label
print("Accessing series with index data like a list: ", pn_series_with_index['c'])



# Some useful functions
print('\n\n\nUseful Functions')
print("------------------------")
print("Size of the data with .shape")
# Size of the series
pn_series_with_index.shape

# Peek at the data
print("Peek at the first 2 records with .head(2)")
print(pn_series_with_index.head(2))

# Can do statistics with data
print("\n\nStatistics")
print("-------------")
series_data = [1, 20, 45, 9, 45]
stats_series = pd.Series(series_data, name="Number of Attendees")
print("Max value: ", stats_series.max())
print("Min Value: ", stats_series.min())
print("Median Value: ", stats_series.median())



# DataFrames
print("\n\n\n\nDataFrames")
print("--------------------")

# Setup Option 1
data_dict = {"Food":['banana', 'apple','orange','bell pepper','kobe beef'], 
             "Costs":[.49, 1.99, 1.39, 1.66, 1200],
             "Cost Unit":["lb","lb","lb","each","oz"]}

food_cost_df = pd.DataFrame(data_dict)
print("Food DataFrame\n\n", food_cost_df)

# Setup Option 2 - From CSV File
data_filepath = "food_data.csv"
food_cost_from_csv_df = pd.read_csv(data_filepath)
print("\n\nFood DataFrame from CSV File\n\n")
print(food_cost_from_csv_df)

# Setup Option 3 - From Exce File
data_filepath = "food_data.xlsx"
food_cost_from_excel_df = pd.read_excel(data_filepath, sheet_name="food_data")
print("\n\nFood DataFrame from CSV File\n\n")
print(food_cost_from_csv_df)


# We have the same functions as we had with Series
print("\n\nDataFrame Functions")
print("-------------------------")
print('DF Shape: ', food_cost_from_csv_df.shape)
print('\n\nHead(2):\n ', food_cost_from_csv_df.head(2))
print('Max Cost: ', food_cost_from_csv_df['Costs'].max())
print("\nRow at index 2:\n", food_cost_from_csv_df.iloc[2])
print('\nColumn Names:\n', food_cost_from_csv_df.columns)


# Filtering values
print("\n\n\nFiltering Datafarmes")
print('---------------------------')
print("Cost of Kobe:\n", food_cost_from_csv_df[food_cost_from_csv_df['Food'] == 'kobe beef'])
print("Cost of bananas and oranges:\n", food_cost_from_csv_df[food_cost_from_csv_df['Food'].isin(['banana','orange'])])
print("Cost is greater than a dollar and less than 2 dollars:\n", food_cost_from_csv_df[(food_cost_from_csv_df['Costs'] > 1) & (food_cost_from_csv_df['Costs'] < 2)])

# Grouping Data
print("Grouping Data")
print('---------------')
print('\nFood Grouped By Cost Unit:\n', food_cost_from_csv_df.groupby('Cost Unit').count())



# Concatenating data
data_dict = {"Food":['banana', 'apple','orange','bell pepper','kobe beef'], 
             "Costs":[.49, 1.99, 1.39, 1.66, 1200],
             "Cost Unit":["lb","lb","lb","each","oz"]}
food_df1 = pd.DataFrame(data_dict)			 
			 
data_dict2 = {"Food":['banana', 'pears','orange','anaheim pepper','chicken breast'], 
             "Costs":[.59, 2.99, 1.99, 3.12, 2.49],
             "Cost Unit":["lb","lb","lb","lb","lb"]}
food_df2 = pd.DataFrame(data_dict2)			 
			 
pd.concat(food_df1, food_df2)


# Mergin/Joiningg Data
new_df = pd.merge(left_df, right_df, how='left', on='col_name')


#dataframe.fillna()
#dataframe.dropna()
#











