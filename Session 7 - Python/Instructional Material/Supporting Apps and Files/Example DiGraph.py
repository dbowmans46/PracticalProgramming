#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:12:56 2023

@author: doug
"""

import networkx
from matplotlib import pyplot as plt


# node_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
node_dict = {'E':'A', 'E':'B', 'E':'C', 'E':'D', 'E':'E'}
node_list = ['A','B','C', 'D', 'E']
label_dict = {'A':'$A$', 'B':'$B$', 'C':'$C$', 'D':'$D$', 'E':'$E$'}

g = networkx.DiGraph()
# bal_tree = networkx.balanced_tree(3,2,g)

for node in node_dict.keys():
    g.add_node(node)

g.add_edge('A','B')
g.add_edge('A','C')
g.add_edge('B','D')
g.add_edge('B','E')

# pos=networkx.spring_layout(g)
pos = {'A':(5,5), 'B':(3,3), 'C':(7,3), 'D':(2,1), 'E':(4,1)}
#pos = {0:(5,5), 1:(3,3), 2:(6,3), 3:(2,1), 4:(4,1)}

# networkx.draw_networkx_nodes(g, pos, nodelist = node_list)
# networkx.draw_networkx_edges(g, pos, edgelist = g.edges)
# networkx.draw_networkx_labels(g, pos, label_dict)

plt.figure(1,figsize=(2,5))
networkx.draw(g, pos)
networkx.draw_networkx_labels(g, pos, label_dict, font_color='#FFFFFF')
plt.show()