# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

plt.figure()

for i in range(len(plotList)):
    plots = plotList[i]
    for plot in plots:
        plt.scatter(i+1, plot*10, c="blue")
    if i==0:
        plt.scatter(i+1, 0.2, c="red", marker="*")
    elif i==1:
        plt.scatter(i+1, 0.4, c="red", marker="*")
    elif i==2:
        plt.scatter(i+1, 0.6, c="red", marker="*")
    elif i==3:
        plt.scatter(i+1, 0.8, c="red", marker="*")
    elif i==4:
        plt.scatter(i+1, 1.0, c="red", marker="*")
    elif i==5:
        plt.scatter(i+1, 1.5, c="red", marker="*")
    elif i==6:
        plt.scatter(i+1, 2.0, c="red", marker="*")
    elif i==7:
        plt.scatter(i+1, 2.5, c="red", marker="*")
    elif i==8:
        plt.scatter(i+1, 3.0, c="red", marker="*")
    elif i==9:
        plt.scatter(i+1, 4.0, c="red", marker="*")
    elif i==10:
        plt.scatter(i+1, 5.0, c="red", marker="*")
    elif i==11:
        plt.scatter(i+1, 7.0, c="red", marker="*")
    elif i==12:
        plt.scatter(i+1, 10.0, c="red", marker="*")
    plt.vlines(i+1, 0, 10,  linestyle="dashed", linewidth=0.5)
plt.show()