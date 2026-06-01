# MIP教程生成执行计划

> 基于 v4 提示词，规划如何用子代理并行生成完整的 19 个文件。

---

## 总体策略

### 为什么不一次性生成
- 单次 LLM 输出上限 ~8,000-16,000 字（中文），而教程总规模 20,000-30,000 字
- 代码文件（7 个 .py）需要在独立上下文中生成，避免混入教程正文
- 并行化可将总耗时从串行 10+ 轮压缩到 4 轮

### 并行拆分原则
- **无依赖的文件 → 并行**：背景知识、求解器技术、各案例之间无内容耦合（每个案例的提示词已自包含）
- **有顺序依赖的 → 串行**：README.md 必须先出（定调 + 术语表），其他文件引用它
- **每对 (案例.md + 案例.py) → 同子代理生成**：保证代码和讲解的一致性

---

## 文件清单（19 个目标文件）

| # | 文件 | 类型 | 预估体量 | 依赖 |
|---|------|------|---------|------|
| 1 | `mip-course/README.md` | 索引 | ~2,000 字 | 无 |
| 2 | `mip-course/code/requirements.txt` | 配置 | 1 行 | 无 |
| 3 | `mip-course/01-background.md` | 教程 | ~5,000 字 | README |
| 4 | `mip-course/02-solver-tech.md` | 教程 | ~6,000 字 | README |
| 5 | `mip-course/03-case-vrp.md` | 案例 | ~3,000 字 | README |
| 6 | `mip-course/code/case01_vrp.py` | 代码 | ~120 行 | 03-case-vrp.md |
| 7 | `mip-course/04-case-scheduling.md` | 案例 | ~3,000 字 | README |
| 8 | `mip-course/code/case02_scheduling.py` | 代码 | ~120 行 | 04-case-scheduling.md |
| 9 | `mip-course/05-case-portfolio.md` | 案例 | ~2,500 字 | README |
| 10 | `mip-course/code/case03_portfolio.py` | 代码 | ~100 行 | 05-case-portfolio.md |
| 11 | `mip-course/06-case-energy.md` | 案例 | ~2,500 字 | README |
| 12 | `mip-course/code/case04_energy.py` | 代码 | ~120 行 | 06-case-energy.md |
| 13 | `mip-course/07-case-network.md` | 案例 | ~3,000 字 | README |
| 14 | `mip-course/code/case05_network.py` | 代码 | ~150 行 | 07-case-network.md |
| 15 | `mip-course/08-capstone.md` | 项目 | ~2,000 字 | README |
| 16 | `mip-course/code/capstone.py` | 代码 | ~100 行 | 08-capstone.md |
| 17 | `mip-course/appendix-a-when-not-mip.md` | 附录 | ~800 字 | 无 |
| 18 | `mip-course/appendix-b-common-pitfalls.md` | 附录 | ~800 字 | 无 |
| 19 | `mip-course/appendix-c-ml-intersection.md` | 附录 | ~800 字 | 无 |
| 20 | `mip-course/appendix-d-reading-list.md` | 附录 | ~500 字 | 无 |

> 实际 20 个文件（README + 7 章 + 4 附录 + 7 代码 + 1 requirements.txt）

---

## 四阶段执行

### Phase 1：我亲自生成（2 个文件，串行）

**原因**：README.md 是整份教程的「地基」——术语表、学习路线、快速导航、环境准备都在里面。所有后续文件都引用它，必须我亲自写以确保一致性。requirements.txt 只有一行，不值得开子代理。

| 步骤 | 文件 | 方法 |
|------|------|------|
| 1.1 | `mip-course/README.md` | 直接 write_file |
| 1.2 | `mip-course/code/requirements.txt` | 直接 write_file |

### Phase 2：第一批子代理（3 个并行子代理，每人 1-2 个文件）

**原因**：01-background 和 02-solver-tech 是两个最大的教程文件（~11,000 字合计），且互相独立。03-case-vrp 是第一个案例也独立。三者并行可省最多时间。

| 子代理 | 文件 | 目标内容 |
|--------|------|---------|
| Agent A | `01-background.md` | 1.1 ~ 1.5（全部背景知识，~5,000 字） |
| Agent B | `02-solver-tech.md` | 2.1 ~ 2.9（全部求解器技术，~6,000 字） |
| Agent C | `03-case-vrp.md` + `code/case01_vrp.py` | 案例1 完整教程 + Python 代码 |

### Phase 3：第二批子代理（3 个并行子代理，每人 2 个文件）

**原因**：案例 2/3/4 互相独立，体量相近（各 ~3,000 字 md + ~120 行 py），非常适合并行。

| 子代理 | 文件 | 目标内容 |
|--------|------|---------|
| Agent A | `04-case-scheduling.md` + `code/case02_scheduling.py` | 案例2：生产排程 |
| Agent B | `05-case-portfolio.md` + `code/case03_portfolio.py` | 案例3：投资组合 (MIQP) |
| Agent C | `06-case-energy.md` + `code/case04_energy.py` | 案例4：发电厂调度 |

### Phase 4：第三批子代理（3 个并行子代理）

**原因**：案例5 是进阶内容较长，毕业项目和附录可以合并处理。

| 子代理 | 文件 | 目标内容 |
|--------|------|---------|
| Agent A | `07-case-network.md` + `code/case05_network.py` | 案例5：电信网络 (Benders) |
| Agent B | `08-capstone.md` + `code/capstone.py` | 毕业项目 + 代码骨架 |
| Agent C | `appendix-a/b/c/d` 四个文件 | 全部附录（~2,900 字合计） |

---

## 子代理上下文设计

每个子代理收到两份信息：
1. **简短指令**：要生成哪些文件、具体要求
2. **v4 提示词中对应的章节规格摘录**：大约 500-1500 字的精简版，只包含该子代理负责部分的约束

### Phase 2 Agent A 上下文示例

```
goal: 生成文件 mip-course/01-background.md

内容要求（摘录自 v4 提示词）：

## 01-background.md — 第一部分：背景知识

包含 1.1 ~ 1.5 全部内容，约 5,000 字。

### 1.1 没有优化之前（略）
[摘录完整的 1.1~1.5 规格]

写作约束：
- 先直觉后公式，每个概念配生活+工作双类比
- 每节末尾加 🆘 逃生通道
- 术语首次出现标注英文
- 公式逐符号解释中文含义

文件格式：
- 开头：<!-- 文件: mip-course/01-background.md -->
- 结尾：> [文件完，下一个: 02-solver-tech.md]
```

### 所有子代理通用约束
```
- 你是运筹优化专家，面向初学者写教程
- 写完自检：奶奶测试 + 术语密度测试
- 代码写到独立的 .py 文件，教程 .md 中引用路径
- .py 文件顶部有依赖声明和 docstring
- .py 文件末尾有 if __name__ == "__main__" 批量测试块
- 禁止裸公式，禁止术语堆砌
```

---

## 子代理需要的工具集

所有子代理使用 `['file']` — 只需要 write_file。不需要 terminal/web/browser。

---

## 验证步骤（Phase 4 完成后）

1. **文件存在性检查**：确认 20 个文件全部生成
   ```bash
   find mip-course -type f | wc -l  # 应为 20
   ```
2. **大小检查**：确认无空文件、无截断
   ```bash
   wc -c mip-course/*.md mip-course/code/*.py
   ```
3. **交叉引用检查**：README 中的快速导航链接指向的文件都存在
4. **代码语法检查**：
   ```bash
   python3 -m py_compile mip-course/code/*.py
   ```
5. **内容完整性抽样**：随机抽取 3 个文件，检查是否包含 🆘 逃生通道、是否公式密度 ≤3/节、是否有双类比

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 子代理输出截断（字数超限） | 中 | 给子代理明确的字数上限 + 「宁可精简不要截断」指令 |
| Phase 2 Agent A/B 输出体量太大 | 中 | 背景和求解器技术可分拆为 Phase 2a/2b（每个文件 2500 字子节） |
| 子代理生成的代码语法错误 | 低 | Phase 5 做语法检查 + 自动修复 |
| 不同子代理风格不一致 | 低 | v4 提示词约束已经足够细粒度 |
| 子代理之间术语翻译不一致 | 低 | README 术语表已锁定，所有子代理需引用 README |

---

## 预估时间

| 阶段 | 并行度 | 预估耗时 |
|------|--------|---------|
| Phase 1 (我) | 1 | ~1 分钟 |
| Phase 2 (3 子代理) | 3 | ~2-3 分钟（最长子代理 ~6000 字） |
| Phase 3 (3 子代理) | 3 | ~2 分钟 |
| Phase 4 (3 子代理) | 3 | ~2 分钟 |
| Phase 5 (验证) | 1 | ~1 分钟 |
| **合计** | | **~8-10 分钟** |

对比串行生成（逐文件 ~12-15 个 LLM 轮次 × 每轮 1-2 分钟）→ 并行可节省 50-60% 时间。

---

## 不委托的部分（亲自做）

1. **README.md** — 教程的「地基」，术语表必须零歧义
2. **requirements.txt** — 一行，不值得开 subagent
3. **Phase 5 验证** — 交叉引用检查、代码语法检查、风格一致性检查
4. **修复** — 子代理输出有任何问题，我亲自修复而非重新委托
