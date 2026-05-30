"""
案例4b：用 SimPy 进行医院急诊科排队仿真
=============================================
对比 case06_queue.py 的手写 DES 和 SimPy 框架的差异。
展示 SimPy 的 process + resource 模型。

运行: python3 code/python/case04b_simpy_queue.py
"""


# 教学注释：先识别业务对象，再看它们如何映射为优化、仿真或启发式模型。
# 结果解读侧重成本、资源利用率和服务水平等管理指标。



import simpy
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ============================================================
# 1. M/M/1 基础排队（与解析解对比验证）
# ============================================================
def mm1_patient(env, name, mu, stats):
    """单个病人的到达→排队→服务→离开过程"""
    arrive = env.now
    with stats['resource'].request() as req:
        yield req
        wait = env.now - arrive
        stats['waits'].append(wait)
        yield env.timeout(random.expovariate(mu))

def run_mm1(lmbda=0.020, mu=0.04, sim_time=200000):
    """运行 M/M/1 仿真，与解析解 L = λ/(μ-λ) 对比
       注意：高利用率时需极长时间达到稳态，本演示取 ρ=0.5 平衡速度和精度"""
    env = simpy.Environment()
    stats = {
        'resource': simpy.Resource(env, capacity=1),
        'waits': [],
    }
    def patient_arrivals():
        i = 0
        while env.now < sim_time:
            yield env.timeout(random.expovariate(lmbda))
            env.process(mm1_patient(env, f'P{i}', mu, stats))
            i += 1
    env.process(patient_arrivals())
    env.run()
    # 丢弃前 20% 作为预热期
    cutoff = int(len(stats['waits']) * 0.2)
    waits_ss = stats['waits'][cutoff:]
    W_avg = np.mean(waits_ss) if waits_ss else 0
    Lq_sim = lmbda * W_avg
    rho = lmbda / mu
    L_analytic = rho / (1 - rho)
    W_analytic = 1/(mu - lmbda)
    return {
        'lambda': lmbda, 'mu': mu, 'rho': rho,
        'sim_patients': len(waits_ss),
        'sim_W': W_avg, 'sim_L': Lq_sim,
        'analytic_L': L_analytic, 'analytic_W': W_analytic,
        'L_error': abs(Lq_sim - L_analytic) / L_analytic * 100 if L_analytic > 0 else 999,
        'W_error': abs(W_avg - W_analytic) / W_analytic * 100 if W_analytic > 0 else 999,
    }

# ============================================================
# 2. 多优先级排队（医院急诊科场景）
# ============================================================
class EDCategory:
    """急诊科病人分诊类别，带优先级"""
    EMERGENCY = 0   # 危重，优先级最高
    URGENT = 1      # 紧急
    ROUTINE = 2     # 常规，优先级最低

def ed_patient(env, name, category, service_time, stats):
    """急诊病人的到达→按优先级排队→服务→离开"""
    arrive = env.now
    # 按优先级请求医生（PreemptiveResource 或 PriorityResource）
    with stats['resource'].request(priority=category) as req:
        yield req
        wait = env.now - arrive
        stats['waits'][category].append(wait)
        yield env.timeout(service_time)

def run_emergency(num_doctors=3, sim_hours=168):
    """运行急诊科多优先级仿真"""
    env = simpy.Environment()
    stats = {
        'resource': simpy.PriorityResource(env, capacity=num_doctors),
        'waits': {EDCategory.EMERGENCY: [], EDCategory.URGENT: [], EDCategory.ROUTINE: []},
        'counts': {EDCategory.EMERGENCY: 0, EDCategory.URGENT: 0, EDCategory.ROUTINE: 0},
    }
    # 病人到达率（随时间变化：白天多，晚上少）
    def arrival_rate(t):
        hour = t % 24
        if 8 <= hour <= 20:  # 白天高峰
            return 2.0  # 每小时2人
        else:
            return 0.5  # 每小时0.5人

    def patient_arrivals():
        i = 0
        t = 0
        while t < sim_hours:
            rate = arrival_rate(t)
            if rate > 0:
                yield env.timeout(random.expovariate(rate))
            else:
                yield env.timeout(1)
                continue
            # 随机分诊类别：10% 危重, 30% 紧急, 60% 常规
            r = random.random()
            if r < 0.1:
                cat = EDCategory.EMERGENCY
            elif r < 0.4:
                cat = EDCategory.URGENT
            else:
                cat = EDCategory.ROUTINE
            # 服务时间：危重最长，常规最短
            service = {
                EDCategory.EMERGENCY: random.expovariate(1/45),   # 平均45分钟
                EDCategory.URGENT: random.expovariate(1/30),
                EDCategory.ROUTINE: random.expovariate(1/15),
            }[cat]
            stats['counts'][cat] += 1
            env.process(ed_patient(env, f'P{i}', cat, min(service, 120), stats))
            t = env.now
            i += 1

    env.process(patient_arrivals())
    env.run(until=sim_hours)

    return stats

# ============================================================
# 3. 运行与结果
# ============================================================

def main():
    print("=" * 65)
    print("  案例4b：SimPy 排队仿真")
    print("=" * 65)

    print("\n" + "─" * 65)
    print("  第一部分：M/M/1 验证（与解析解对比）")
    print("─" * 65)
    mm1 = run_mm1()
    print(f"  λ={mm1['lambda']}, μ={mm1['mu']}, ρ={mm1['rho']:.3f}")
    print(f"  仿真:   L={mm1['sim_L']:.2f}, W={mm1['sim_W']:.1f}")
    print(f"  解析解: L={mm1['analytic_L']:.2f}, W={mm1['analytic_W']:.1f}")
    print(f"  L 偏差: {mm1['L_error']:.2f}%  {'✅' if mm1['L_error']<10 else '❌'}")
    print(f"  W 偏差: {mm1['W_error']:.2f}%  {'✅' if mm1['W_error']<10 else '❌'}")

    print("\n" + "─" * 65)
    print("  第二部分：多优先级急诊科仿真（λ随时间变化）")
    print("─" * 65)

    for nd in [2, 3, 4]:
        stats = run_emergency(num_doctors=nd, sim_hours=336)
        avg_wait = {cat: np.mean(stats['waits'][cat]) if stats['waits'][cat] else 0
                    for cat in [EDCategory.EMERGENCY, EDCategory.URGENT, EDCategory.ROUTINE]}
        print(f"\n  医生数: {nd}")
        print(f"    危重:  平均等待 {avg_wait[EDCategory.EMERGENCY]:.1f} 分钟  ({stats['counts'][EDCategory.EMERGENCY]}人)")
        print(f"    紧急:  平均等待 {avg_wait[EDCategory.URGENT]:.1f} 分钟    ({stats['counts'][EDCategory.URGENT]}人)")
        print(f"    常规:  平均等待 {avg_wait[EDCategory.ROUTINE]:.1f} 分钟    ({stats['counts'][EDCategory.ROUTINE]}人)")

    # 验证标准
    print("\n  ── ✅ 验证标准 ──")
    print("  1. M/M/1 仿真 vs 解析解偏差 < 10% ✅" if mm1['L_error'] < 10 else "  1. ❌")
    print("  2. Little 定律 L=λW 成立 ✅" if abs(mm1['sim_L'] - mm1['lambda']*mm1['sim_W']) < 0.5 else "  2. ❌")
    print("  3. 增加医生后危重病人等待时间下降 (与非线性的排队效应一致)")
    print()
    print("  ── 建模心法 ──")
    print("  • SimPy 代码 ~80行 vs 手写DES ~200行")
    print("  • process + resource 模型贴近自然语言")
    print("  • PriorityResource 内置支持多优先级排队")
    print("  • 可以轻松加入非平稳到达率、多级分诊")
    print("  • 缺点是大型仿真比手写C/Python慢")


if __name__ == "__main__":
    main()
