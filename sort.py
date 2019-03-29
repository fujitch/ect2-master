# -*- coding: utf-8 -*-

import numpy as np

indexes = np.argsort(output[:, 0])

plotList = []

for i in range(len(indexes)):
    index = indexes[i]
    if i == 0:
        plots = []
    elif output[index, 0] != output[indexes[i-1], 0]:
        plotList.append(plots)
        plots = []
    plots.append(out[index, 0])
plotList.append(plots)