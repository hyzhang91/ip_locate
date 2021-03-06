# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 16:48:33 2018

@author: Hyzhang
"""

import copy
import itertools
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
 
T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
 
 
def draw_point(data):
    X, Y = [], []
    for p in data:
        X.append(p[0])
        Y.append(p[1])
    plt.plot(X, Y, 'bo')
 
 
def draw_line(xy_list):
    for xy in xy_list:
        x, y = xy
        plt.plot(x, y, 'g', lw=2)
 
 
def draw_square(square_list):
    currentAxis = plt.gca()
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y", '#EB70AA', '#0099FF'])
    for square in square_list:
        currentAxis.add_patch(
            Rectangle((square[0][0], square[0][1]), square[1][0] - square[0][0], square[1][1] - square[0][1],
                      color=next(colors)))
 
 
def median(lst):
    m = len(lst) / 2
    return lst[m], m
 
 
history_quare = []
 
 
def build_kdtree(data, d, square):
    history_quare.append(square)
    data = sorted(data, key=lambda x: x[d])
    p, m = median(data)
 
    del data[m]
    print data, p
 
    if m >= 0:
        sub_square = copy.deepcopy(square)
        if d == 0:
            sub_square[1][0] = p[0]
        else:
            sub_square[1][1] = p[1]
        history_quare.append(sub_square)
        if m > 0: build_kdtree(data[:m], not d, sub_square)
    if len(data) > 1:
        sub_square = copy.deepcopy(square)
        if d == 0:
            sub_square[0][0] = p[0]
        else:
            sub_square[0][1] = p[1]
        build_kdtree(data[m:], not d, sub_square)
 
 
build_kdtree(T, 0, [[0, 0], [10, 10]])
print history_quare
 
 
# draw an animation to show how it works, the data comes from history
# first set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], 'g', lw=2)
label = ax.text([], [], '')
 
# initialization function: plot the background of each frame
def init():
    plt.axis([0, 10, 0, 10])
    plt.grid(True)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('build kd tree (www.hankcs.com)')
    draw_point(T)
 
 
currentAxis = plt.gca()
colors = itertools.cycle(["#FF6633", "g", "#3366FF", "c", "m", "y", '#EB70AA', '#0099FF', '#66FFFF'])
 
# animation function.  this is called sequentially
def animate(i):
    square = history_quare[i]
    currentAxis.add_patch(
        Rectangle((square[0][0], square[0][1]), square[1][0] - square[0][0], square[1][1] - square[0][1],
                  color=next(colors)))
    return
 
# call the animator.  blit=true means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history_quare), interval=1000, repeat=False,
                               blit=False)
plt.show()
anim.save('kdtree_build.gif', fps=2, writer='imagemagick')