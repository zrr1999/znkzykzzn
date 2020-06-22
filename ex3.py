# ex3.py
from algorithm import Swarm, MasterSwarm, clip
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False


def response(t, z=0.707):
    sz = np.sqrt(1 - z ** 2)
    beta = np.arccos(z)
    return 1 - np.exp(-z * t) / 0.7072 * sz * np.sin(t*sz + beta)


def cal_ts(z, dt=0.01):
    # 估算上界
    z_inv = 1 / z
    ts1 = z_inv * 3.5
    ts2 = -z_inv * (np.log(0.05) + np.log(1 - z ** 2) / 2)

    sz = np.sqrt(1 - z ** 2)
    beta = np.arccos(z)
    ts = ts2
    for t in np.arange(ts2, 0, -0.01):
        y_t = np.exp(-z * t) / sz * np.sin(t*sz + beta)
        if y_t < -0.05 or y_t > 0.05:
            ts = t + 0.01
            break
    return ts, ts1, ts2


def fit_function(z):
    out = cal_ts(z)[0]
    return -out


# 搜寻响应最大值
# s = Swarm(response, [[0], [4 * 3.14], [-10], [10]], 10)  # 创建粒子群
# fg = plt.figure()
# plt.ion()
# for epoch in range(10):
#     plt.cla()
#     plt.xlabel("$t$")
#     plt.ylabel("$y(t)$")
#     plt.xlim(-5, 10)
#     plt.ylim(-5, 10)
#
#     s.step()
#     fitness = s.fit()
#     s.render(c='r')
#
#     print(fitness, s.best_position)
#     plt.pause(0.1)
    # plt.show()
    # if epoch % 2 == 0:
    #     fg.savefig(f"resources/3_{epoch//2+1}")

# 打印三曲线
fg = plt.figure()
zeta = np.arange(0.1, 1, 0.01)
ts = np.array([cal_ts(z) for z in zeta])
plt.plot(zeta, ts[:, 0], c="red", label='真实调整时间')
plt.plot(zeta, ts[:, 1], label='近似公式')
plt.plot(zeta, ts[:, 2], c="blue", label='包络线调整时间')
plt.legend()
fg.savefig(f"resources/4_0")
plt.show()
# 搜寻最优zeta值
s = Swarm(fit_function, [[0.1], [0.9], [-10], [10]], 20)  # 创建粒子群
fg = plt.figure()
for epoch in range(10):

    plt.cla()
    plt.xlabel(r"$\zeta$")
    plt.ylabel("$-y_s$")
    plt.xlim(0, 1)
    plt.ylim(-30, 0)
    s.render()
    plt.plot(zeta, -ts[:, 0], c="red")

    plt.pause(0.1)
    if epoch % 2 == 0:
        fg.savefig(f"resources/4_{epoch//2+1}")

    s.step()
    fitness = s.fit()
    print(fitness, s.best_position)

plt.ioff()
plt.show()
