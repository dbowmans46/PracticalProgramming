#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:22:43 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Data wrangling tools
import pandas as pd

# Plotting tools
import matplotlib.pyplot as plt


# Fetch example
# The data is used to attempt to categorize topics of messages.
from sklearn.datasets import fetch_20newsgroups
newsgroup_data = fetch_20newsgroups()

# For this example, we will create a historgram of the count of each type of
# target.  First, wrangle data and get counts.
newsgroup_df = pd.DataFrame(data = newsgroup_data.target, columns=["Target"])
newsgroup_df["Target Name"] = newsgroup_df.apply(lambda row: newsgroup_data.target_names[row[0]], axis=1)
grouped_data = newsgroup_df.groupby("Target Name",axis=0).count()

# Make the histogram
plt.bar(grouped_data.index, grouped_data['Target'].values, width=0.8, align='center')
plt.title("Count of News Piece Topics in Scikit-Learn fetch_20newsgroups")
plt.xticks(rotation=310, ha='left')


# Load Example
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# For this example, we will try and print 5 dimensions on a 3D scatter plot
# First, data wrangle
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df = pd.DataFrame(iris_dataset.data, columns=feature_names)
iris_df['Targets'] = iris_dataset.target

# Setup the 3D plot.
fig, ax = plt.subplots(
    figsize=(100,100),
    facecolor="white",
    tight_layout=True,
    subplot_kw={"projection": "3d"},
)

standard_text_size = 150
fig.suptitle("Iris Data", size=200)
# Plot each data point, separating out each classification with its own
# data point marker
plot_markers = ['o', '^', 's']
for target, data in iris_df.groupby('Targets'):
    test = ax.scatter(data['sepal_length'], data['sepal_width'], \
                data['petal_length'], c=data['petal_width'], \
                marker=plot_markers[target], label=plot_markers[target], s=10000)
plt.legend(iris_dataset.target_names, prop={'size': standard_text_size})

# Make the view a little easier to see everything
ax.view_init(azim=-45, elev=30)

cbar = fig.colorbar(test, ax=ax, orientation="horizontal", shrink=0.8, aspect=60, pad=0.01)
cbar.ax.tick_params(labelsize=standard_text_size)
cbar.ax.set_xlabel('Petal Width (cm)', size=standard_text_size)

ax.set_xlabel('Sepal Length (cm)', size=standard_text_size)
ax.set_ylabel('Sepal Width (cm)', size=standard_text_size)
ax.set_zlabel('Petal Length (cm)', size=standard_text_size)
plt.show()


# Make example
# We have already seen this one from the LLE example.  This time, we will
# make the points finer.
from sklearn.datasets import make_swiss_roll
points, color = make_swiss_roll(n_samples=200000, noise=0, random_state=0)

x, y, z = points.T

fig, ax = plt.subplots(
    figsize=(100, 100),
    facecolor="white",
    tight_layout=True,
    subplot_kw={"projection": "3d"},
)

fig.suptitle("Swiss Roll", size=160)
col = ax.scatter(x, y, z, c=color, s=50, alpha=0.6)
ax.view_init(azim=-60, elev=9)
fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
plt.show()