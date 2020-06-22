import os
import re

from pylatex import Command, Figure, NewLine, NoEscape, Itemize

from pytex import Core, DocTree
from pytex.markdown import md2tex
from pytex.pseudocode import algorithm, al_function, al_if, al_for
from pytex.utils import array

packages = [["geometry", "a4paper,centering,scale=0.8"], "amsmath", "graphicx", "amssymb", "bm"]
core = Core(packages=packages, debug=True)

define = [[r"\dif"], [r"\text{d}"]]
core.global_define(*define)

core.pre_append(title=Command('heiti', '智能控制与控制智能课程作业报告'),
                author=Command('kaishu', '六个骨头'),
                date=Command('today'))

core.body_append(Command('maketitle'))
core.body_append(Command('tableofcontents'))

goal = "充实生活"  # 目的
sh = ""  # 收获
ref = (r"\begin{thebibliography}{1}"  # 参考文献
       r"\bibitem{zhihu} 如何直观形象地理解粒子群算法？https://www.zhihu.com/question/23103725/answer/365298309"
       r"\end{thebibliography}")

al, alc = algorithm("粒子群算法", "适应度函数", "最优值", core=core, label=["算法", "输入", "输出"])
func = al_function("Particle_Swarm_Optimization", "适应度函数")
func.add_state(NoEscape("初始化参数"))
func.add_state(NoEscape("初始化粒子位置"))
alf = al_for("1:10")
alf.append("sdas")
func.add_state(alf)
func.add_state(NoEscape("计算粒子群的速度并计算位置"))
func.add_state(Command("Return", "最优值"))
alc.append(func)

al2, alc = algorithm("改进粒子群算法", "适应度函数", "最优值", core=core, label=["算法", "输入", "输出"])
func = al_function("Particle_Swarm_Optimization", "适应度函数")
func.add_state(NoEscape("初始化参数"))
func.add_state(NoEscape("初始化粒子位置"))
func.add_state(Command("Return", "最优值"))
alc.append(func)

env = Itemize()
env.add_item("系统：Windows 10")
env.add_item("开发工具：PyCharm、VSCode")
env.add_item("编程语言：Python3.7")
env.add_item("Python库：Copy(用于深拷贝), Numpy(用于计算粒子速度和位置), Matplotlib(用于可视化),, mpl_toolkits(用于3D可视化)")
env.add_item("报告编写语言：LaTex（使用PyTex编写生成、XeLaTex编译）")

fg = Itemize()  # 分工
fg.add_item(NoEscape("蚁群算法设计：詹荣瑞、安然、刘欢"))
fg.add_item(NoEscape("实验一分析：刘欢"))
fg.add_item(NoEscape("实验二分析：安然"))
fg.add_item(NoEscape("实验三分析：詹荣瑞"))
fg.add_item(NoEscape("报告编写：詹荣瑞、刘欢、安然"))

# 粒子群算法介绍
lzq = (r"粒子群算法(Particle Swarm Optimization,简称PSO)是1995年Eberhart博士和Kennedy博士一起提出的。"
       r"粒子群算法是通过模拟鸟群捕食行为设计的一种群智能算法。"
       r"区域内有大大小小不同的食物源，鸟群的任务是找到最大的食物源（全局最优解）。"
       r"鸟群在整个搜寻的过程中，通过相互传递各自位置的信息，让其他的鸟知道食物源的位置最终，"
       r"整个鸟群都能聚集在食物源周围，即我们所说的找到了最优解，问题收敛。学者受自然界的启发开发了诸多类似智能算法，"
       r"如蚁群算法、布谷鸟搜索算法、鱼群算法、捕猎算法等等。",

       "粒子群算法的目标是使所有粒子在多维超体（multi-dimensional hyper-volume）中找到最优解。"
       "首先给空间中的所有粒子分配初始随机位置和初始随机速度。"
       "然后根据每个粒子的速度、问题空间中已知的最优全局位置和粒子已知的最优位置依次推进每个粒子的位置。"
       "随着计算的推移，通过探索和利用搜索空间中已知的有利位置，粒子围绕一个或多个最优点聚集或聚合。"
       "该算法设计玄妙之处在于它保留了最优全局位置和粒子已知的最优位置两个信息。"
       "后续的实验发现，保留这两个信息对于较快收敛速度以及避免过早陷入局部最优解都具有较好的效果。"
       "这也奠定了后续粒子群算法改进方向的基础。",
       [al], al2)

ex1 = (r"绘制单变量正态分布在区间$[-4,4]$上的波形$p(x)\sim N(0, 1)$，并利用粒子群优化算法求解其\textbf{最大值}。"
       )
ex2 = (r"绘制二元正态分布在区间$([-4,4]，[-4,4])$上波形$p(x_1，x_2 )\sim N(\bm{\mu}, \Sigma)$，"
       r"并利用粒子群优化算法求解其\textbf{最大值}。"
       f"已知条件：$\\bm{{\\mu}}={array([[0], [0]]).dumps()},\\Sigma={array([[1, 0], [0, 1]]).dumps()}$"
       )
ex3 = (r"典型二阶欠阻尼控制系统若干结论验证。"
       r"查阅相关文献，利用粒子群优化算法求解$t$在$[0,4\pi]$区间二阶欠阻尼系统单位阶跃响应的\textbf{最大值}"
       r"（假设$\omega_n=1\text{rad/s}, \zeta=0.707$），计算此时系统的\textbf{超调量}，"
       r"讨论\textbf{粒子群大小}和\textbf{最大迭代次数}对寻优结果的\textbf{影响}。"
       r"编程绘制出误差带$\Delta=5\%$时，阻尼比$\zeta$（在区间$0\leq\zeta\leq1$）"
       r"与调整时间$t_s$之间的\textbf{关系曲线}（三条关系曲线，真实调整时间、包络线调整时间、近似公式）；"
       r"利用粒子群优化算法，以真实调整时间$t_s$作为粒子群优化算法的适应度函数"
       r"当误差带为$\Delta=5\%$时，优化得到$0\leq\zeta\leq1$区间内的最优$\zeta$值，绘制出\textbf{收敛曲线}；"
       )

# 实验分析
fx = (r"单变量正态分布的概率密度函数"
      r"\[ p(x)=\frac {1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x-\mu )^2}{2\sigma^2}} \]"
      r"代入题目数据得到："
      r"\[ p(x)=\frac {1}{\sqrt{2 \pi} } e^{-\frac{x^2}{2}} \]",

      r"多元随机变量正态分布的概率密度函数"
      r"\[ p(x)=\frac {1}{\sqrt{(2 \pi)^n|\Sigma|} } e^{-\frac 1 2 (x-\mu )^T \Sigma^{-1}(x-\mu )} \]"
      r"代入题目数据得到："
      r"\[ p(x)=\frac {1}{2 \pi} e^{-\frac 1 2 x^T x} \]",

      r"典型二阶控制系统微分方程一般形式：\[\frac {\dif^2y(t)}{\dif t^2}+2\zeta "
      r"\omega_n\frac {\dif y(t)}{\dif t}+\omega_n^2y(t)=\omega_n^2f(t)\]"
      r"欠阻尼$0<\zeta<1$时，其单位阶跃响应为："
      r"\[ y(t)=1-\frac {e^{-\zeta \omega_n t}}{\sqrt{1-\zeta^2} } \text{sin}({\omega_d t +\beta}) \]"
      r"代入题目数据得到："
      r"\[ y(t)=1-\frac {e^{-0.707 t}}{0.7072} \text{sin}({0.7072 t +0.7855}) \]"
      r"以$y(t)$为适应度函数使用粒子群算法得到最大值$y(4.44)\approx1.043$，此时超调量$\sigma\%\approx 4.3\%$",
      )

doc_tree = DocTree({
    "title": "实验内容",
    "content": [{
        "title": "实验 1",
        "content": NoEscape(ex1)
    }, {
        "title": "实验 2",
        "content": NoEscape(ex2)
    }, {
        "title": "实验 3",
        "content": NoEscape(ex3)
    }]
}, {
    "title": "实验环境及成员分工",
    "content": [{
        "title": "实验环境",
        "content": env
    }, {
        "title": "实验分工",
        "content": fg
    }]
}, {
    "title": "实验目的",
    "content": NoEscape(goal)
}, {
    "title": "实验原理",
    "content": {
        "title": "粒子群算法",
        "content": [{
            "title": "简介",
            "content": lzq[0]
        }, {
            "title": "算法策略",
            "content": lzq[1]
        }, {
            "title": "算法实现",
            "content": lzq[2]
        }, {
            "title": "算法改进",
            "content": lzq[3]
        }]
    }
}, {
    "title": "实验分析",
    "content": [{
        "title": "实验 1",
        "content": NoEscape(fx[0])
    }, {
        "title": "实验 2",
        "content": NoEscape(fx[1])
    }, {
        "title": "实验 3",
        "content": NoEscape(fx[2])
    }]
}, {
    "title": "实验步骤",
    "content": "内容"
}, {
    "title": "体会与收获",
    "content": NoEscape(sh)
}, packages=al.packages)
core.body_append(doc_tree, NoEscape(ref))

print("正在生成pdf")
core.generate_pdf('resources/work', compiler='XeLatex', clean_tex=False)
print("生成完成！\n欢迎使用PyTex！")
