# ex2.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
from algorithm import Swarm, MasterSwarm


def plot():
    """
    绘制二元正态分布波形图

    :return: None
    """
    num = 200
    lines = np.linspace(-4, 4, num)
    X, Y = np.meshgrid(lines, lines)

    u = np.array([0, 0])  # 均值
    o = np.array([[1, 0], [0, 1]])  # 协方差矩阵

    pos = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)  # 定义坐标点

    a = np.dot((pos - u), np.linalg.inv(o))  # o的逆矩阵
    b = np.expand_dims(pos - u, axis=3)
    # Z = np.dot(a.reshape(200*200,2),(pos-u).reshape(200*200,2).T)
    Z = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        Z[i] = [np.dot(a[i, j], b[i, j]) for j in range(num)]  # 计算指数部分

    Z = np.exp(Z * (-0.5)) / (2 * np.pi * np.sqrt(np.linalg.det(o)))

    fig = plt.figure()  # 绘制图像
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6, cmap=cm.coolwarm)

    ax.contour(X, Y, Z, 10, zdir='z', offset=0, cmap=cm.coolwarm)  # 绘制xy面投影
    ax.contour(X, Y, Z, zdir='x', offset=-4, cmap=mpl.cm.winter)  # 绘制zy面投影
    ax.contour(X, Y, Z, zdir='y', offset=4, cmap=mpl.cm.winter)  # 绘制zx面投影

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def fit_function(x):
    x = np.array(x)
    k = 1 / (2 * np.pi)
    return k * np.exp(-1 / 2 * np.dot(x.T, x))


if __name__ == '__main__':
    # 绘制二元正态分布波形图
    plot()
    # 粒子群算法求最值
    s1 = Swarm(fit_function, [[-1, -1], [1, 1]])  # 创建粒子群
    s2 = MasterSwarm(fit_function, [[-1, -1], [1, 1]])
    plt.figure()
    plt.ion()
    for epoch in range(100):
        plt.cla()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        fitness1 = s1.fit()
        s1.step()
        s1.render(c="b")

        fitness2 = s2.fit()
        s2.step()
        s2.render(c='r')
        print(fitness1, fitness2, s1.best_position, s2.best_position)
        plt.pause(0.1)
    plt.ioff()
    plt.show()
