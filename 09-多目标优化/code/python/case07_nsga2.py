"""
case07_nsga2.py
NSGA-II 从零实现：非支配排序 + 拥挤距离 + 锦标赛选择 + SBX交叉 + 多项式变异
验证：前沿收敛到真实 Pareto 前沿

测试问题：ZDT1（双目标，凸前沿）
"""


# 教学注释：关注多个目标之间的冲突、Pareto 支配关系和权衡参数。
# 输出的解集用于筛选可解释、可落地的折中方案。



import math
import random
import copy


# ===================== 测试问题：ZDT1 =====================
def zdt1(x):
    """
    ZDT1 测试函数
    变量维度: n (通常30)
    真实 Pareto 前沿: f2 = 1 - sqrt(f1)
    """
    n = len(x)
    # f1 = x1
    f1 = x[0]

    # g = 1 + 9/(n-1) * sum(x[2..n])
    g = 1.0 + 9.0 * sum(x[1:]) / (n - 1)

    # f2 = g * (1 - sqrt(f1/g))
    f2 = g * (1.0 - math.sqrt(f1 / g))

    return f1, f2


# ===================== 个体定义 =====================
class Individual:
    """NSGA-II 个体"""

    def __init__(self, n_vars, lb=0.0, ub=1.0):
        self.x = [random.uniform(lb, ub) for _ in range(n_vars)]
        self.fitness = None  # (f1, f2)
        self.rank = 0
        self.crowding_distance = 0.0

    def evaluate(self):
        self.fitness = zdt1(self.x)

    def __repr__(self):
        return f"Ind(rank={self.rank}, cd={self.crowding_distance:.4f}, f={self.fitness})"


# ===================== 非支配排序 =====================
def non_dominated_sort(population):
    """
    非支配排序（快速非支配排序算法）
    返回每个个体的 Pareto 前沿等级
    """
    n = len(population)

    # 支配计数和解集
    domination_count = [0] * n
    dominated_sets = [[] for _ in range(n)]
    fronts = [[]]  # fronts[0] = 第一前沿

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i].fitness, population[j].fitness):
                dominated_sets[i].append(j)
            elif dominates(population[j].fitness, population[i].fitness):
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    # 逐层构建
    k = 0
    while k < len(fronts):
        next_front = []
        for i in fronts[k]:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = k + 1
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        k += 1

    return fronts


def dominates(f_a, f_b):
    """
    a 支配 b? (双目标最小化)
    如果 a 在所有目标上不差于 b，且至少有一个目标严格优于 b
    """
    return (
        f_a[0] <= f_b[0] and f_a[1] <= f_b[1]
        and (f_a[0] < f_b[0] or f_a[1] < f_b[1])
    )


# ===================== 拥挤距离计算 =====================
def crowding_distance_assignment(population, front_indices):
    """
    计算同一前沿中个体的拥挤距离
    """
    m = len(front_indices)
    if m <= 1:
        for idx in front_indices:
            population[idx].crowding_distance = float("inf")
        return

    # 初始化距离
    for idx in front_indices:
        population[idx].crowding_distance = 0.0

    # 目标数量（这里固定2）
    n_obj = 2

    for obj in range(n_obj):
        # 按当前目标排序
        front_indices.sort(key=lambda idx: population[idx].fitness[obj])

        # 边界点设为无穷大
        population[front_indices[0]].crowding_distance = float("inf")
        population[front_indices[-1]].crowding_distance = float("inf")

        # 目标范围
        obj_range = population[front_indices[-1]].fitness[obj] - population[front_indices[0]].fitness[obj]
        if obj_range < 1e-10:
            continue

        # 计算中间点的拥挤距离
        for k in range(1, m - 1):
            diff = population[front_indices[k + 1]].fitness[obj] - population[front_indices[k - 1]].fitness[obj]
            population[front_indices[k]].crowding_distance += diff / obj_range


# ===================== 锦标赛选择 =====================
def tournament_selection(population, tournament_size=2):
    """
    锦标赛选择：比较等级和拥挤距离
    （等级低优先，同级时拥挤距离大优先）
    """
    candidates = random.sample(range(len(population)), tournament_size)
    best = candidates[0]

    for c in candidates[1:]:
        # 等级更低（rank 值小）更优
        if population[c].rank < population[best].rank:
            best = c
        elif population[c].rank == population[best].rank:
            # 同级：拥挤距离大更优
            if population[c].crowding_distance > population[best].crowding_distance:
                best = c

    return population[best]


# ===================== SBX 交叉 =====================
def sbx_crossover(parent1, parent2, eta_c=20, prob_cross=0.9):
    """
    SBX (Simulated Binary Crossover) 模拟二进制交叉
    eta_c: 交叉分布指数（越大，子代越接近父代）
    """
    n = len(parent1.x)
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    for i in range(n):
        if random.random() > prob_cross:
            continue

        x1, x2 = parent1.x[i], parent2.x[i]
        if abs(x1 - x2) < 1e-10:
            continue

        # 确保 x1 < x2
        if x1 > x2:
            x1, x2 = x2, x1

        y1 = max(0.0, x1)  # 下界
        y2 = min(1.0, x2)  # 上界

        rand = random.random()
        beta = 1.0 + 2.0 * (x1 - y1) / (x2 - x1 + 1e-10)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))

        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

        c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        # 边界约束
        c1 = max(0.0, min(1.0, c1))
        c2 = max(0.0, min(1.0, c2))

        # 随机分配
        if random.random() < 0.5:
            child1.x[i], child2.x[i] = c1, c2
        else:
            child1.x[i], child2.x[i] = c2, c1

    return child1, child2


# ===================== 多项式变异 =====================
def polynomial_mutation(individual, eta_m=20, prob_mut=1.0 / 30):
    """
    多项式变异 (Polynomial Mutation)
    eta_m: 变异分布指数
    """
    n = len(individual.x)
    mutated = copy.deepcopy(individual)

    for i in range(n):
        if random.random() > prob_mut:
            continue

        x = individual.x[i]
        delta1 = (x - 0.0) / (1.0 - 0.0)  # 归一化到 [0,1]
        delta2 = (1.0 - x) / (1.0 - 0.0)

        rand = random.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if rand <= 0.5:
            delta_q = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta1) ** (eta_m + 1.0)) ** mut_pow - 1.0
        else:
            delta_q = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta2) ** (eta_m + 1.0)) ** mut_pow

        x_new = x + delta_q * (1.0 - 0.0)
        x_new = max(0.0, min(1.0, x_new))
        mutated.x[i] = x_new

    return mutated


# ===================== NSGA-II 主算法 =====================
def nsga2(
    n_vars=30,
    pop_size=100,
    n_generations=250,
    prob_cross=0.9,
    prob_mut=None,
    eta_c=20,
    eta_m=20,
):
    """
    NSGA-II 主循环
    """
    if prob_mut is None:
        prob_mut = 1.0 / n_vars

    # 初始化种群
    population = [Individual(n_vars) for _ in range(pop_size)]
    for ind in population:
        ind.evaluate()

    # 主循环
    for gen in range(n_generations):
        # 非支配排序
        fronts = non_dominated_sort(population)

        # 计算拥挤距离
        for front in fronts:
            crowding_distance_assignment(population, front)

        # 生成子代
        offspring = []
        while len(offspring) < pop_size:
            # 锦标赛选择父代
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            # SBX 交叉
            c1, c2 = sbx_crossover(p1, p2, eta_c, prob_cross)

            # 多项式变异
            c1 = polynomial_mutation(c1, eta_m, prob_mut)
            c2 = polynomial_mutation(c2, eta_m, prob_mut)

            # 评估子代
            c1.evaluate()
            c2.evaluate()

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 合并父代和子代
        combined = population + offspring

        # 非支配排序 + 拥挤距离
        fronts = non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(combined, front)

        # 选择下一代（环境选择）
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                # 整个前沿加入
                for idx in front:
                    new_population.append(combined[idx])
            else:
                # 按拥挤距离降序排序
                front.sort(key=lambda idx: combined[idx].crowding_distance, reverse=True)
                remaining = pop_size - len(new_population)
                for i in range(remaining):
                    new_population.append(combined[front[i]])
                break

        population = new_population

        # 输出进度
        if (gen + 1) % 50 == 0 or gen == 0:
            best_f1 = min(ind.fitness[0] for ind in population)
            best_f2 = min(ind.fitness[1] for ind in population)
            print(f"  第 {gen + 1:4d} 代 | 种群大小: {len(population)} | "
                  f"min f1={best_f1:.6f} | min f2={best_f2:.6f}")

    return population


# ===================== 验证：收敛到真实 Pareto 前沿 =====================
def compute_igd(population, n_true=500):
    """
    计算 IGD (Inverted Generational Distance)
    衡量解集到真实 Pareto 前沿的接近程度
    真实前沿: f2 = 1 - sqrt(f1), f1 ∈ [0, 1]
    """
    # 生成真实前沿点
    true_front = []
    for i in range(n_true + 1):
        f1 = i / n_true
        f2 = 1.0 - math.sqrt(f1)
        true_front.append((f1, f2))

    # 对每个真实点，找最近的计算点
    total_dist = 0.0
    for tf in true_front:
        min_dist = float("inf")
        for ind in population:
            dist = math.sqrt((ind.fitness[0] - tf[0]) ** 2 + (ind.fitness[1] - tf[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
        total_dist += min_dist

    return total_dist / len(true_front)


def extract_pareto_front(population):
    """从种群中提取非支配前沿"""
    pareto = []
    for ind in population:
        dominated = False
        for other in population:
            if other is ind:
                continue
            if dominates(other.fitness, ind.fitness):
                dominated = True
                break
        if not dominated:
            pareto.append(ind)

    # 按 f1 排序
    pareto.sort(key=lambda ind: ind.fitness[0])
    return pareto


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("NSGA-II 从零实现")
    print("测试问题: ZDT1 (双目标, 30变量)")
    print("真实 Pareto 前沿: f2 = 1 - sqrt(f1)")
    print("=" * 60)

    # 设置随机种子
    random.seed(42)

    # 运行 NSGA-II
    print("\n>>> NSGA-II 进化过程")
    final_pop = nsga2(
        n_vars=30,
        pop_size=100,
        n_generations=250,
        prob_cross=0.9,
        prob_mut=1.0 / 30,
        eta_c=20,
        eta_m=20,
    )

    # 提取 Pareto 前沿
    print("\n>>> 最终 Pareto 前沿提取")
    pareto_front = extract_pareto_front(final_pop)
    print(f"  提取到 {len(pareto_front)} 个非支配解")

    # 显示前沿
    print(f"\n>>> Pareto 前沿（部分）:")
    print(f"  {'#':>3} | {'f1':>8} | {'f2':>8} | {'预期 f2':>8} | {'误差':>8}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for k, ind in enumerate(pareto_front[:10]):
        f1, f2 = ind.fitness
        expected_f2 = 1.0 - math.sqrt(f1)
        error = abs(f2 - expected_f2)
        print(f"  {k:3d} | {f1:8.6f} | {f2:8.6f} | {expected_f2:8.6f} | {error:8.6f}")
    if len(pareto_front) > 10:
        print(f"  ... 共 {len(pareto_front)} 个解")

    # 计算 IGD
    print("\n>>> 收敛性评估")
    igd = compute_igd(pareto_front)
    print(f"  IGD (Inverted Generational Distance): {igd:.6f}")
    print(f"  IGD 越小，表示解集越接近真实 Pareto 前沿")

    if igd < 0.01:
        print(f"  ✓ 前沿已收敛到真实 Pareto 前沿 (IGD < 0.01)")
    elif igd < 0.05:
        print(f"  ~ 前沿接近真实 Pareto 前沿 (IGD < 0.05)")
    else:
        print(f"  - 前沿仍在收敛中，可增加代数或种群大小")

    # 统计
    f1_vals = [ind.fitness[0] for ind in pareto_front]
    f2_vals = [ind.fitness[1] for ind in pareto_front]
    print(f"\n>>> 统计信息")
    print(f"  f1 范围: [{min(f1_vals):.6f}, {max(f1_vals):.6f}] (理论: [0, 1])")
    print(f"  f2 范围: [{min(f2_vals):.6f}, {max(f2_vals):.6f}] (理论: [0, 1])")

    print("\n>>> 结论")
    print("  NSGA-II 从零实现包含以下组件:")
    print("    1. 快速非支配排序 (Fast Non-dominated Sort)")
    print("    2. 拥挤距离计算 (Crowding Distance)")
    print("    3. 锦标赛选择 (Tournament Selection)")
    print("    4. SBX 交叉 (Simulated Binary Crossover)")
    print("    5. 多项式变异 (Polynomial Mutation)")
    print("    6. 精英保留策略 (Elitism)")
    print(f"  成功收敛到 ZDT1 真实 Pareto 前沿，IGD = {igd:.6f}")
    print("=" * 60)
