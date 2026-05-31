# 项目审查报告 — 运筹学与优化方法课程系列

> 审查日期：2026-05-31
> 审查人：Hermes Agent
> 状态：所有建议均未执行，待确认后逐一修改

## 审查结论

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码审查 | 7.5/10 | 代码质量好，缺运行测试 |
| 安全审计 | 9.5/10 | 几乎完美，无硬编码凭证或危险调用 |
| 重构建议 | 6.0/10 | 公共代码重复，无 CI |
| 内容优化 | 8.0/10 | 文档质量高，缺交叉引用和课程 README |
| **综合** | **7.75/10** | |

## 问题清单（12 项）

### 🔴 高优（4 项）

#### 1. 补全运行测试
- **文件**：`run_checks.py`（顶层）+ 9 门课各缺 `code/python/run_checks.py`
- **问题**：仅课程 10 有自己的 `run_checks.py`，顶层只做 `py_compile` 语法检查不执行代码
- **方案**：将顶层 `compile_python()` 改为运行各课程 `capstone.py`；为每门课复制课程 10 的 `run_checks.py` 模板

#### 2. 提取公共数学库
- **文件**：新建 `common/stats_utils.py`，修改 6 个现有文件
- **问题**：`normal_cdf`/`normal_pdf`/`normal_ppf`/`generate_normal` 在 6 个文件中重复实现
  - `01-概率论/case03_distributions.py`
  - `01-概率论/case04_clt.py`
  - `01-概率论/case06_hypothesis.py`
  - `01-概率论/case07_monte_carlo.py`
  - `01-概率论/capstone.py`
  - `07-库存供应链/case04_newsvendor.py`
- **方案**：取课程 07 的实现（有类型提示+极值处理），创建 `common/stats_utils.py`，其余文件改为 `from common.stats_utils import ...`

#### 3. 课程间交叉引用
- **文件**：10 门课的末尾章节 `.md`
- **问题**：全项目仅 8 处跨课程引用，课程间彼此独立
- **方案**：每门课末章加 "延伸阅读" 段落，链接到相关课程的对应章节
- **推荐矩阵**：
  - 01-概率论 → 05(应用), 08(随机规划)
  - 02-算法 → 04(MIP), 05(运筹)
  - 04-MIP → 05(运筹), 09(多目标)
  - 05-运筹 → 02(算法), 04(MIP), 06(博弈), 07(供应链)
  - 07-供应链 → 01(概率), 08(随机)

#### 4. 每门课加 README.md
- **文件**：10 个课程目录各新建 `README.md`
- **问题**：新读者进入课程目录不知道该从哪开始
- **方案**：统一模板：课程目标 + 前置知识 + 文件导航 + 运行命令

### 🟡 中优（4 项）

#### 5. 补齐 10-凸优化的教学文档
- **文件**：`10-凸优化与非线性优化/` 下新建 4-5 个 `.md`
- **问题**：有 7 个 Python 文件，但只有 3 个教学文档
- **方案**：按代码文件对应补齐（梯度下降、投资组合、逻辑回归、约束设计、非凸陷阱）

#### 6. 加 GitHub Actions CI
- **文件**：新建 `.github/workflows/smoke.yml`
- **问题**：无自动测试
- **方案**：push/PR 时自动 `pip install -r requirements.txt && make test`

#### 7. requirements.txt 补充 highspy
- **文件**：`requirements.txt`
- **问题**：缺少 `highspy`，`gurobipy` 未标为可选
- **方案**：加 `highspy`，`gurobipy` 加注释说明为商业求解器

#### 8. docstring 语言统一
- **文件**：`10-凸优化/code/python/case04_portfolio_qp.py:1`
- **问题**：仅此文件用英文 docstring，其余 61 个文件均用中文
- **方案**：改为中文

### 🟢 低优（4 项）

#### 9. 全局随机种子移入 main()
- **文件**：`05-运筹学/code/python/` 下 6 个文件
- **问题**：`random.seed(42)` 和 `np.random.seed(42)` 在模块顶级执行，import 时污染全局状态
- **方案**：移到各文件 `main()` 函数第一行

#### 10. shapiro_wilk p 值修正
- **文件**：`01-概率论/case03_distributions.py:156`
- **问题**：`return w, w` — p 值直接等于 W 统计量，无统计意义
- **方案**：改为 `return w, None` 并注释"精确 p 值需查表"

#### 11. C++ 编译产物清理
- **文件**：`run_checks.py:40`
- **问题**：`compile_cpp()` 写文件到 `/tmp/` 后不删除
- **方案**：加 `finally: output.unlink(missing_ok=True)`

#### 12. Makefile 扩展
- **文件**：`Makefile`
- **问题**：缺少 `make test` 和 `make cpp` 目标
- **方案**：加两个目标，`test` 运行每门课的 `run_checks.py`，`cpp` 编译所有 C++ 文件
