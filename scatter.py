# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

plt.figure()

for i in range(len(plotList)):
    plots = plotList[i]
    for plot in plots:
        plt.scatter(i+1, plot*0.5, c="blue")
    if i==0:
        plt.scatter(i+1, 0.001, c="yellow", marker="*")
    elif i==1:
        plt.scatter(i+1, 0.002, c="yellow", marker="*")
    elif i==2:
        plt.scatter(i+1, 0.005, c="yellow", marker="*")
    elif i==3:
        plt.scatter(i+1, 0.01, c="yellow", marker="*")
    elif i==4:
        plt.scatter(i+1, 0.02, c="yellow", marker="*")
    elif i==5:
        plt.scatter(i+1, 0.05, c="yellow", marker="*")
    elif i==6:
        plt.scatter(i+1, 0.1, c="yellow", marker="*")
    elif i==7:
        plt.scatter(i+1, 0.2, c="yellow", marker="*")
    elif i==8:
        plt.scatter(i+1, 0.5, c="yellow", marker="*")
    plt.vlines(i+1, 0, 0.6,  linestyle="dashed", linewidth=0.5)
plt.show()