# algorithm.py
import numpy as np
import matplotlib.pyplot as plt
import copy


class Particle(object):
    """粒子
    储存粒子信息，包括粒子最优位置、最优适应度。
    """
    def __init__(self, fit_func, domain, inertia=1, social_learning=2,
                 individual_learning=2, learning_rate=0.1):
        """

        :param fit_func: 适应度函数，用于处理粒子的位置参数，输出粒子的适应度
        :param domain: 粒子位置的定义域
        :param inertia: 惯性系数，类似于质量和转动惯量
        :param social_learning: 社交系数，体现出粒子随大流的意愿
        :param individual_learning: 独立系数，体现出粒子自信的程度
        :param learning_rate: 学习率，类似于时间微分，可以理解为每一步过去的时间
        """
        self.domain = np.array(domain)
        self.fit_func = fit_func
        self.position = self.domain[0]+np.random.rand(self.domain.shape[1])*(self.domain[1]-self.domain[0])
        self.fitness = self.fit_func(self.position)  #
        self.velocity = 0
        self.best_position = copy.deepcopy(self.position)  # pbest
        self.best_fitness = copy.deepcopy(self.fitness)  #
        self.alpha = learning_rate

        self.w = inertia
        self.c1 = social_learning
        self.c2 = individual_learning

    def fit(self):
        """
        计算并储存粒子当前位置的的最优适应度(没有更优位置不更新)。

        :return: 粒子的最优适应度
        """
        self.fitness = self.fit_func(self.position)
        if self.fitness > self.best_fitness:
            self.best_fitness = copy.deepcopy(self.fitness)
            self.best_position = copy.deepcopy(self.position)
        return self.fitness

    def step(self, g_best):
        """
        计算粒子的速度并移动。

        :param: g_best: 粒子群的最优位置
        :return: None
        """
        self.velocity *= self.w
        self.velocity += self.c1 * (self.best_position - self.position)*np.random.rand()
        self.velocity += self.c2 * (g_best - self.position)*np.random.rand()
        if self.domain.shape[0] == 4:
            self.velocity = clip(self.velocity, self.domain[2], self.domain[3])
        else:
            self.velocity = clip(self.velocity, self.domain[0], self.domain[1])

        self.position += self.alpha*self.velocity
        self.position = clip(self.position, self.domain[0], self.domain[1])


class Swarm(object):
    """粒子群
    储存粒子群信息，包括粒子群最优位置、最优适应度，同时储存了所有粒子。
    """
    def __init__(self, fit_func, domain, swarm_size=100, particle_class=Particle):
        """

        :param fit_func: 适应度函数，用于处理粒子的位置参数，输出粒子的适应度
        :param domain: 粒子位置的定义域[[min...], [max...]]
        :param swarm_size: 粒子群大小，即粒子的数目
        :param particle_class: 粒子类型，可自己实现具备其它功能的粒子
        """
        self.domain = np.array(domain)
        self.particles = [particle_class(fit_func, self.domain) for _ in range(swarm_size)]
        self.best_position = self.particles[0].best_position  # gbest
        self.best_fitness = self.particles[0].best_fitness  #

    def fit(self):
        """
        计算并储存粒子群当前位置的的最优适应度(没有更优位置不更新)。

        :return: 粒子群的最优适应度
        """
        for particle in self.particles:
            particle.fit()
            if self.best_fitness < particle.best_fitness:
                self.best_position = particle.best_position
                self.best_fitness = particle.best_fitness
        return self.best_fitness

    def step(self):
        """
        计算每个粒子的速度并移动。

        :return: None
        """
        for particle in self.particles:
            particle.step(self.best_position)

    def render(self, continuous=True, c=None):
        """
        打印粒子群各粒子到散点图。

        :param continuous: 暂无
        :param c: 粒子颜色
        :return: None
        """
        pos = np.array([particle.position for particle in self.particles])
        fit = np.array([particle.fitness for particle in self.particles])
        if len(self.best_position) == 1:
            plt.scatter(pos, fit, s=10, c=c)
            plt.scatter(self.best_position, self.best_fitness, s=80, c=c)
        else:
            plt.scatter(pos[:, 0], pos[:, 1], s=10, c=c)
            plt.scatter(*self.best_position, s=80, c=c)


class MasterSwarm(Swarm):
    def __init__(self, fit_func, domain, master_size=10, slave_size=10, particle_class=Particle):
        size = master_size*slave_size
        super().__init__(fit_func, domain, size, particle_class)
        self.master_size = master_size
        self.master_num = np.arange(0, size+1, slave_size)
        self.slave_position = [self.particles[i].best_position for i in self.master_num[:-1]]  # gbest
        self.slave_fitness = [self.particles[i].best_fitness for i in self.master_num[:-1]]  #

    def fit(self, slave=True):
        for i in range(self.master_size):
            for particle in self.particles[self.master_num[i]: self.master_num[i+1]]:
                particle.fit()
                if self.slave_fitness[i] < particle.best_fitness:
                    self.slave_position[i] = particle.best_position
                    self.slave_fitness[i] = particle.best_fitness
            if self.best_fitness < self.slave_fitness[i]:
                self.best_position = self.slave_position[i]
                self.best_fitness = self.slave_fitness[i]
        return self.best_fitness

    def step(self):
        for i in range(self.master_size):
            for j, particle in enumerate(self.particles[self.master_num[i]: self.master_num[i+1]]):
                if j == 0:
                    particle.step(self.best_position)
                else:
                    particle.step(self.slave_position[i])


def clip(x, min_value, max_value):
    return np.maximum(np.minimum(x, max_value), min_value)


if __name__ == '__main__':
    # 限制了最大速度
    def fit_function(x):
        out = []
        for i, ix in enumerate(x[1:]):
            out.append(abs(ix-x[i]))
        return -sum(out)
    s1 = Swarm(fit_function, [[0, 0], [1, 1], [-10, -10], [10, 10]])  # 创建粒子群
    s2 = MasterSwarm(fit_function, [[0, 0], [1, 1], [-10, -10], [10, 10]])
    plt.figure()
    plt.ion()

    for epoch in range(1000):
        plt.cla()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        fitness1 = s1.fit()
        s1.step()
        s1.render(c="b")

        fitness2 = s2.fit()
        s2.step()
        s2.render(c='r')
        plt.pause(0.1)
        print(fitness1, fitness2, s1.best_position, s2.best_position)
    plt.ioff()
    plt.show()
