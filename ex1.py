# ex1.py
from algorithm import Swarm
import numpy as np
import matplotlib.pyplot as plt


def fit_function(x):
    k = 1 / (2 * np.pi)
    return k * np.exp(-1 / 2 * x**2)


s = Swarm(fit_function, [[-10], [10], [-100], [100]])  # 创建粒子群
plt.figure()
plt.ion()

for epoch in range(1000):
    plt.cla()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    s.step()
    fitness = s.fit()
    s.render(c="b")
    plt.pause(0.1)
    print(fitness, s.best_position)
plt.ioff()
plt.show()
