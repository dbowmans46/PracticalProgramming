#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:17:55 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from sklearn.datasets import make_s_curve
import matplotlib as mpl
import matplotlib.pyplot as plt

# Code modified from SkLearn's website, https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
# and Hands-On ML GitHub page: https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb
points, color = make_s_curve(n_samples=1000, noise=0, random_state=0)

x, y, z = points.T

fig, ax = plt.subplots(
    figsize=(6, 6),
    facecolor="white",
    tight_layout=True,
    subplot_kw={"projection": "3d"},
)
fig.suptitle("S-Curve", size=16)
col = ax.scatter(x, y, z, c=color, s=50, alpha=0.8)
ax.view_init(azim=-60, elev=9)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax.zaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
plt.show()


from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(points)

# Code to generate Figure 8–12. Unrolled Swiss roll using LLE:

plt.title("Unrolled S curve using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

#save_fig("lle_unrolling_plot")
plt.show()