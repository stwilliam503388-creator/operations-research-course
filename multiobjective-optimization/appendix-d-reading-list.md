# 附录D：推荐阅读

> 以下资源按难度排序，从入门到进阶。覆盖教材、开源工具、经典论文、课程视频和博客文章五个维度。

---

## 教材

| 书名 | 适合 | 评价 |
|------|------|------|
| 《Multi-Objective Optimization Using Evolutionary Algorithms》(Coello Coello) | 入门~进阶 | 经典教材，NSGA-II 作者之一，代码齐全，案例丰富——**想买一本就买这本** |
| 《多目标进化优化》(郑金华) | 入门 | 中文经典，写得清楚，案例丰富，国内高校多目标课程的常用教材 |
| 《MCDM: Past Decade and Future Trends》(Zanakis et al.) | 进阶 | 综述型，了解 MCDM（多准则决策）全景的好读物 |
| 《Evolutionary Algorithms for Multi-Objective Optimization》(Deb) | 进阶 | **NSGA-II 作者原典**，理论深入但代码少——适合想深入研究算法原理的人 |
| 《多目标决策分析》(岳超源) | 入门 | 偏 MCDM 方向，理论清晰，适合运筹学背景的读者 |

**一句话选书指南**：想要 Python 代码 → Coello Coello。想要中文 → 郑金华。想要深度理论 → Deb。想要决策分析视角 → 岳超源。

---

## 开源工具

| 工具 | 语言 | 说明 |
|------|------|------|
| [pymoo](https://pymoo.org) | Python | **强烈推荐**。多目标优化库，含 NSGA-II / MOEA/D / NSGA-III / SMS-EMOA 等 50+ 算法，文档完善，API 设计优雅 |
| [PlatEMO](https://github.com/BIMK/PlatEMO) | MATLAB | 多目标优化平台，含 100+ 算法和测试函数，学术界的标准比较平台 |
| [DEAP](https://deap.readthedocs.io) | Python | 进化算法框架，灵活但需要自己实现多目标组件（非支配排序等）——适合想动手造轮子的人 |
| [Optuna](https://optuna.org) | Python | 超参数调优工具，支持多目标优化（TPE + NSGA-II）——如果你的场景是 AutoML 多目标，这是首选 |
| [jMetal](https://jmetal.github.io/jMetal) | Java | 成熟的 Java 多目标优化框架，适合企业级 Java 技术栈 |
| [pygmo](https://esa.github.io/pygmo2/) | Python | 并行计算友好的多目标优化库，支持自定义算法和问题 |

---

## 经典论文

| 论文 | 年份 | 为什么读 |
|------|------|---------|
| Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" | 2002 | NSGA-II 原始论文，引用 50,000+，**多目标领域最重要的单篇论文** |
| Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition" | 2007 | MOEA/D 原始论文，分解式方法开创之作 |
| Zitzler et al., "SPEA2: Improving the Strength Pareto Evolutionary Algorithm" | 2001 | SPEA2 原始论文，SPEA 系列改进版 |
| Zitzler & Thiele, "Multiobjective Evolutionary Algorithms: A Comparative Case Study" | 1999 | 经典对比论文，奠定了性能评价框架的基础 |
| Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints" (NSGA-III) | 2014 | 高维多目标（3+ 目标）的解决方案——NSGA-III |

---

## 课程与视频

| 资源 | 说明 |
|------|------|
| **本课程** | 多目标优化基础课程——**你正在读的这个** |
| Prof. Kalyanmoy Deb 的系列讲座 (YouTube) | NSGA-II / NSGA-III 作者的讲座，从理论基础到前沿进展 |
| Coursera "Multi-Objective Optimization" (NPTEL) | 印度理工学院的免费课程，偏理论但讲解清晰 |
| "Multi-Objective Optimization" by Prof. Carlos Coello Coello (YouTube) | 另一位重量级人物的讲座系列 |

---

## 博客与文章

| 资源 | 说明 |
|------|------|
| pymoo 官方文档中的教程 | 从入门到实战的最快路径（比论文好读 10 倍）|
| "A Tutorial on Multi-Objective Optimization" (Medium / Towards Data Science) | 搜这个关键词能发现多篇不错的入门文章 |
| PlatEMO 的 Algorithm List | 想了解有哪些多目标算法→直接看 PlatEMO 的算法列表（100+ 种）|

---

## 一句话总结

> **如果你只想读一本书：Coello Coello 的《Multi-Objective Optimization Using Evolutionary Algorithms》**
>
> **如果你只想用一个库：pymoo (Python)**
>
> **如果你只想记住一件事：多目标优化不是找一个最优解，而是找一组最好的折衷**

---

> 回到目录 [README.md](README.md)
