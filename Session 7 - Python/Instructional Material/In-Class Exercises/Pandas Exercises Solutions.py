# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:18:54 2021

@author: Doug
"""
# Series In-Class Exercises
# TODO: Create series with data [1, 2, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10]
data_points = [1, 2, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10]
import pandas as pd
series_data = pd.Series(data_points)

# TODO: Set the name to "data points"
series_data.name = 'data points'

# TODO: Get the maximum, median, mode, and minimum values
print("Maximum: ", series_data.max())
print("Median: ", series_data.median())
print("Mode: ", series_data.mode())
print("Minimum: ", series_data.min())

# TODO: How many rows are there?  Programmatically determine.
num_rows = series_data.shape[0]
print("Rows: ", num_rows)

# TODO: Print the first 4 records
print("First 4 rows:\n")
print(series_data.head(4))



# DataFrame In-Class Exercises

# TODO: Load the example orbital mechanics dataset
orbital_mechanics_sheet = "Orbital Mechanics"
worksheet_filepath = "Orbital Elements.xlsx"
orbital_mechanics_df = pd.read_excel(worksheet_filepath, orbital_mechanics_sheet)

# TODO: Load the example planet symbols dataset
symbols_sheet = "Planet Symbols"
planet_symbols_df = pd.read_excel(worksheet_filepath, symbols_sheet)

# TODO: Combine the planet symbol data to the orbital mechanics data
combined_data_df = pd.merge(orbital_mechanics_df, planet_symbols_df, on="Planet")


# TODO: Filter the data so only the inner planets remain
combined_data_df = combined_data_df[combined_data_df["Planet"].isin(['Mercury','Venus','Earth','Mars'])]

