# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:18:54 2021

@author: Doug
"""
# Series In-Class Exercises
# TODO: Create series with data [1, 2, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10]
# TODO: Set the name to "data points"
# TODO: Get the maximum, median, mode, and minimum values
# TODO: How many rows are there?  Programmatically determine.
# TODO: Print the first 4 records




# DataFrame In-Class Exercises

# TODO: Load the example orbital mechanics dataset from Orbital Elements.xlsx into a DataFrame
import pandas as pd
data_filepath = 'Orbital Elements.xlsx'
orbital_sheet_name = 'Orbital Mechanics'
orbit_df = pd.read_excel(data_filepath, sheet_name=orbital_sheet_name)

# TODO: Load the example planet symbols dataset from Orbital Elements.xlsx into a DataFrame
symbols_sheet_name = 'Planet Symbols'
planet_symbols_df = pd.read_excel(data_filepath, sheet_name=1)

# TODO: Add the planet symbol data as a new column appended to the orbital
# mechanics data.  Put the data into a new DataFrame.
combined_data_df = pd.merge(orbit_df, planet_symbols_df, how='left', on='Planet')

# TODO: Filter the data so only the inner planets remain
inner_planets_df = combined_data_df[combined_data_df["Planet"].isin(['Mercury','Venus','Earth','Mars'])]
#inner_planets_df = combined_data_df[0:4]
#inner_planets_df = combined_data_df.iloc[0:4]

# TODO: Add Pluto's data: 
# Planet, Semimajor Axis (AU), Eccentricity, Orbital Periods (Earth Years), Mass (Earth Masses),Number of Known Satellites, Symbol
# Pluto, 39.4821, 0.24883 , 248.0208, 0.00220, 5, ♇
# TODO: Get all planets that are more massive than Earth (Pluto can be included at your discretion)
# TODO: Add a column that gives the mass of each planet in kg's.  Earth is roughly 5.9742E24 kg.
# TODO: Reset the index so it is the planet name