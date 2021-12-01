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
worksheet_filepath = "../Data/Orbital Elements.xlsx"
orbital_mechanics_df = pd.read_excel(worksheet_filepath, orbital_mechanics_sheet)

# TODO: Load the example planet symbols dataset
symbols_sheet = "Planet Symbols"
planet_symbols_df = pd.read_excel(worksheet_filepath, symbols_sheet)

# TODO: Combine the planet symbol data to the orbital mechanics data
combined_data_df = pd.merge(orbital_mechanics_df, planet_symbols_df, on="Planet")


# TODO: Filter the data so only the inner planets remain
inner_planets_df = combined_data_df[combined_data_df["Planet"].isin(['Mercury','Venus','Earth','Mars'])]
#inner_planets_df = combined_data_df[0:4]
#inner_planets_df = combined_data_df.iloc[0:4]

# TODO: Add Pluto's data to the original DataFrame containing all data: 
# Planet, Semimajor Axis (AU), Eccentricity, Orbital Periods (Earth Years), Mass (Earth Masses),Number of Known Satellites, Symbol
# Pluto, 39.4821, 0.24883 , 248.0208, 0.00220, 5, ♇
pluto_data = {'Planet':'Pluto', 
              'Semimajor Axis (AU)':39.4821, 
              'Eccentricity':0.24883 , 
              'Orbital Periods (Earth Years)':248.0208, 
              'Mass (Earth Masses)':0.00220, 
              'Number of Known Satellites':5, 
              'Symbol':'♇'}
combined_data_df = combined_data_df.append(pluto_data, ignore_index=True)

# TODO: Get all planets that are more massive than Earth (Pluto can be included at your discretion)
print(combined_data_df[combined_data_df['Mass (Earth Masses)'] > combined_data_df.iloc[2]['Mass (Earth Masses)']])

# TODO: Add a column that gives the mass of each planet in kg's.  Earth is roughly 5.9742E24 kg.
combined_data_df['Mass (kg)'] = combined_data_df['Mass (Earth Masses)']*5.9742E24

# TODO: Reset the index so it is the planet name
combined_data_df = combined_data_df.set_index(combined_data_df['Planet'])
combined_data_df = combined_data_df.drop('Planet', axis=1)

