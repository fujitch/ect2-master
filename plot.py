# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import numpy as np

a = dataset[0]
a = a[6:]
aa = np.zeros((93))
"""
for k in range(93):
    aa[k] = a[k] * (random.random() * 0.2 + 0.9)
a = aa
"""
a = a.reshape(31, 3)

cmap = plt.get_cmap("tab10")

# plt.plot(range(31), a[:, 0], label="Vr")
# plt.plot(range(31), a[:, 1], label="Vi", color=cmap(1))
plt.plot(range(31), a[:, 2], label="|V|", color=cmap(2))
plt.legend(fontsize=20)

plt.show()