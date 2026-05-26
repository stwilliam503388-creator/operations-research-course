<!-- 文件: pde-course/appendix-d-reading-list.md -->

# 附录D：推荐阅读

> 这篇附录列的不是「所有 PDE 相关的书」——那太多了。列的是：如果你读完了这份教程还想深入，从哪里开始最自然，每一本/份资源解决什么问题。

---

## 入门级（读得进去，不劝退）

- **《Numerical Solution of Partial Differential Equations》— Gordon D. Smith**（150 页小册子，用最简单的方式讲 FDM。没有 FEM，没有复杂理论。最适合 FDM 入门。）

- **《Computational Physics》— Mark Newman**（物理系的数值计算教材。不专门讲 PDE，但热传导、波动、拉普拉斯方程的 FDM 代码都有。代码清晰，可以直接运行。）

- **《A First Course in the Numerical Analysis of Differential Equations》— Arieh Iserles**（剑桥教材，数学上严谨但保持可读。如果你想知道「稳定性分析到底在干什么」，这本书是最好的选择。）

## 进阶级（工程实战）

- **《Numerical Heat Transfer and Fluid Flow》— Suhas V. Patankar**（传热和流体的圣经 FVM 教材。Sinha 在 1980 年写的小册子，至今仍是 FVM 实现最清晰的标准教材。如果你要做传热/流体仿真，这本绕不开。）

- **《Finite Element Procedures》— Klaus-Jürgen Bathe**（FEM 的经典工程教材。厚重的 1000+ 页，但结构清晰。不需要全读——结构力学和传热的 FEM 实现章节非常实用。）

- **《Computational Fluid Dynamics: The Basics with Applications》— John D. Anderson**（CFD 入门。如果你要从头建一个 CFD 求解器，这本书的心理模型非常清晰。）

## 理论背景（理解「为什么」）

- **《Partial Differential Equations》— Lawrence C. Evans**（研究生级别的 PDE 理论教材。不教你写代码，但帮你理解 PDE 的「本质」——解的存在性、唯一性、正则性。如果你在地铁上想读点深刻的，读 Evans。）

- **《Partial Differential Equations for Scientists and Engineers》— Stanley J. Farlow**（一句话总结每一类 PDE 的物理背景和求解方法。像字典一样好用。）

## 前沿交叉（PDE + ML）

- **"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems" — Raissi, Perdikaris & Karniadakis (2019)**（PINN 的开山之作。引用量 10000+，不是因为它最完美，而是因为它第一个证明了这个方向可行。）

- **"Fourier Neural Operator for Parametric Partial Differential Equations" — Li et al. (2021)**（FNO 的论文。如果你想知道 Neural Operator 在干什么，这是必读。）

- **《Scientific Machine Learning》— 网上公开笔记 / MIT 18.337**（本科/研究生水平的科学 ML 课程笔记，覆盖 PINN、Neural Operator、概率数值方法等。）

## 工具与代码（边读边跑）

- **FEniCS** (fenicsproject.org) — Python/C++ 的 FEM 库。安装有点麻烦，但用起来是工程级体验。
- **scikit-fem** (github.com/kinnala/scikit-fem) — Python 轻量 FEM 库。比 FEniCS 易装，适合教学。
- **Dedalus** (dedalus-project.org) — 谱方法 PDE 求解器。适合偏微分方程的理论研究。
- **JAX** (jax.readthedocs.io) + **JAX-CFD** — 自动微分 + PDE 求解的现代工具栈。适合 PINN 和 Neural Operator 实验。

## 优先级排序

```
1. Gordon D. Smith — FDM 入门       [如果时间只够读一本]
2. Patankar — FVM 传热与流体        [如果你做传热/CFD]
3. Raissi et al. (2019) — PINN     [如果你对 ML+PDE 感兴趣]
4. FEniCS 教程 — FEM 工程实战       [如果你需要 FEM]
```

> 最重要的不是读了多少，而是多少读到能跑出代码。每个资源都配有示例代码——找出来，跑一遍，改一改。**读不如写，写不如跑。**

> [附录D完]
