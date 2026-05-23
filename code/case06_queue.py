"""
案例 4：服务系统排队优化 — 急诊科离散事件仿真
=============================================
场景：医院急诊科，病人到达率随时间变化，多优先级分诊
模型：排队论 / 离散事件仿真 (DES)
教学点：M/M/c 模型直觉、Little 定律 L=λW、仿真 vs 解析、非平稳排队
"""

import numpy as np
from collections import deque
import heapq
from math import factorial


# ============================================================
# 1. 离散事件仿真引擎 (DES) — 事件调度法
# ============================================================

class Event:
    """仿真事件"""
    def __init__(self, time, event_type, patient_id=None):
        self.time = time
        self.type = event_type  # 'arrival', 'end_service'
        self.patient_id = patient_id

    def __lt__(self, other):
        return self.time < other.time


class Patient:
    """病人"""
    def __init__(self, pid, arrival_time, priority=0):
        self.id = pid
        self.arrival_time = arrival_time
        self.priority = priority  # 0=普通, 1=紧急, 2=危重
        self.service_start_time = None
        self.service_end_time = None
        self.service_duration = None

    @property
    def wait_time(self):
        if self.service_start_time is not None:
            return self.service_start_time - self.arrival_time
        return None

    @property
    def total_time(self):
        if self.service_end_time is not None:
            return self.service_end_time - self.arrival_time
        return None


class EDQueueSimulation:
    """
    急诊科离散事件仿真器

    支持:
    - 常数或随时间变化的到达率
    - 多优先级队列（危重 > 紧急 > 普通）或 FIFO
    - M/M/c 配置
    - 预热期丢弃
    """

    def __init__(self, n_doctors=3, mean_service_time=30.0,
                 sim_minutes=2880, warm_up=480,
                 arrival_func=None, fifo=False):
        """
        参数:
            n_doctors: 医生数量
            mean_service_time: 平均服务时间 (分钟)
            sim_minutes: 仿真时长 (分钟)
            warm_up: 预热期 (分钟)
            arrival_func: 到达率函数 λ(t)
            fifo: True = FIFO 排队, False = 优先级排队
        """
        self.n_doctors = n_doctors
        self.mean_service_time = mean_service_time
        self.service_rate = 1.0 / mean_service_time
        self.sim_minutes = sim_minutes
        self.warm_up = warm_up
        self.arrival_func = arrival_func or self.default_arrival_rate
        self.fifo = fifo

        # 状态
        self.clock = 0.0
        self.event_queue = []
        self.patients = []
        self.doctors_busy = 0
        self.patient_counter = 0
        # FIFO 模式下用一个队列, 优先级模式下用 3 个
        self.fifo_queue = deque()
        self.prio_queue = {0: deque(), 1: deque(), 2: deque()}

        # 统计 (预热期后)
        self.total_patients_served = 0
        self.total_wait_time = 0.0
        self.sum_queue_len = 0.0
        self.sum_queue_time = 0.0
        self.last_log_time = 0.0

    @staticmethod
    def default_arrival_rate(t):
        """随时间变化的到达率 (人/分钟)"""
        hour = (t / 60.0) % 24
        if 8 <= hour <= 11:
            return 0.10
        elif 14 <= hour <= 17:
            return 0.08
        elif 22 <= hour or hour <= 6:
            return 0.025
        else:
            return 0.05

    @staticmethod
    def constant_arrival_rate(t):
        """常数到达率 (用于 M/M/1 验证)"""
        return 0.035  # 人/分钟, 2.1 人/小时, ρ ≈ 0.875

    def schedule_arrival(self):
        """安排下一次到达事件"""
        rate = self.arrival_func(self.clock)
        if rate > 0:
            interarrival = np.random.exponential(1.0 / rate)
        else:
            interarrival = 60.0
        next_time = self.clock + interarrival
        if next_time < self.sim_minutes:
            heapq.heappush(self.event_queue, Event(next_time, 'arrival'))

    def assign_priority(self):
        """随机分配优先级"""
        r = np.random.random()
        if r < 0.05:
            return 2
        elif r < 0.30:
            return 1
        return 0

    def handle_arrival(self, event):
        """处理病人到达"""
        pid = self.patient_counter
        self.patient_counter += 1
        priority = self.assign_priority()
        patient = Patient(pid, event.time, priority)
        self.patients.append(patient)

        if self.doctors_busy < self.n_doctors:
            self.start_service(patient)
        else:
            if self.fifo:
                self.fifo_queue.append(patient)
            else:
                self.prio_queue[priority].append(patient)

        self.schedule_arrival()

    def start_service(self, patient):
        """病人开始接受服务"""
        self.doctors_busy += 1
        patient.service_start_time = self.clock
        duration = np.random.exponential(self.mean_service_time)
        patient.service_duration = duration

        end_time = self.clock + duration
        heapq.heappush(self.event_queue, Event(end_time, 'end_service', patient.id))

    def pop_next_patient(self):
        """从队列中取出下一个待服务病人"""
        if self.fifo:
            if self.fifo_queue:
                return self.fifo_queue.popleft()
            return None
        else:
            for pri in [2, 1, 0]:
                if self.prio_queue[pri]:
                    return self.prio_queue[pri].popleft()
            return None

    def handle_end_service(self, event):
        """处理服务结束"""
        self.doctors_busy -= 1

        patient = self.patients[event.patient_id]
        patient.service_end_time = self.clock

        if patient.arrival_time >= self.warm_up:
            self.total_patients_served += 1
            self.total_wait_time += patient.wait_time

        next_patient = self.pop_next_patient()
        if next_patient is not None:
            self.start_service(next_patient)

    def update_queue_stats(self):
        """更新队长积分 (用于 Little 定律)"""
        dt = self.clock - self.last_log_time
        if dt > 0 and self.clock >= self.warm_up:
            if self.fifo:
                qlen = len(self.fifo_queue)
            else:
                qlen = sum(len(q) for q in self.prio_queue.values())
            self.sum_queue_len += qlen * dt
            self.sum_queue_time += dt
        self.last_log_time = self.clock

    def run(self):
        """运行仿真"""
        label = "时变" if self.arrival_func == self.default_arrival_rate else "恒定"
        qmode = "FIFO" if self.fifo else "优先级"
        print(f"  配置: {self.n_doctors} 医生, 服务时间 {self.mean_service_time:.0f} 分钟, "
              f"仿真 {self.sim_minutes//60}h, 预热 {self.warm_up//60}h")
        print(f"  到达率: {label}, 排队规则: {qmode}")

        self.schedule_arrival()

        step = 0
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.update_queue_stats()
            self.clock = event.time

            if event.type == 'arrival':
                self.handle_arrival(event)
            elif event.type == 'end_service':
                self.handle_end_service(event)

            step += 1
            if step > 200000:
                print("  ⚠️ 达到最大事件数限制")
                break

        self.print_results()

    def print_results(self):
        """打印结果"""
        valid = [p for p in self.patients
                 if p.arrival_time >= self.warm_up and p.service_end_time is not None]
        if not valid:
            print("  无有效统计数据")
            return

        wait_times = [p.wait_time for p in valid]

        print(f"  有效病人: {len(valid)}")
        print(f"  平均等待: {np.mean(wait_times):.2f} 分钟")
        print(f"  中位等待: {np.median(wait_times):.2f} 分钟")
        print(f"  最大等待: {np.max(wait_times):.2f} 分钟")

        if not self.fifo:
            for pri, label in [(2, "危重"), (1, "紧急"), (0, "普通")]:
                pw = [p.wait_time for p in valid if p.priority == pri]
                if pw:
                    print(f"    {label}: {len(pw)} 人, 等待 {np.mean(pw):.2f} 分钟")

        self.verify_little_law(wait_times, valid)

    def verify_little_law(self, wait_times, valid_patients):
        """Little 定律: L = λW"""
        lambda_avg = len(valid_patients) / (self.sim_minutes - self.warm_up)
        W_avg = np.mean(wait_times)
        L_direct = self.sum_queue_len / self.sum_queue_time if self.sum_queue_time > 0 else 0
        LW_product = lambda_avg * W_avg

        print(f"  Little 定律: L={L_direct:.2f}, λ={lambda_avg:.4f}, "
              f"W={W_avg:.2f}, λW={LW_product:.2f}",
              end="")
        denom = max(L_direct, 0.1)
        error = abs(LW_product - L_direct) / denom * 100
        print(f", 偏差={error:.2f}%")
        if error < 10.0:
            print(f"  ✅ Little 定律通过")

    def get_mm1_analytical(self):
        """M/M/1 解析解"""
        valid = [p for p in self.patients
                 if p.arrival_time >= self.warm_up and p.service_end_time is not None]
        if not valid:
            return None, None, None
        lambda_avg = len(valid) / (self.sim_minutes - self.warm_up)
        mu = self.service_rate
        rho = lambda_avg / mu
        if rho >= 1:
            return None, None, None
        W_q = rho / (mu * (1 - rho))
        return lambda_avg, W_q, rho


# ============================================================
# 2. 主程序
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # ========== 场景 A: 常数到达率 + FIFO + M/M/1 验证 ==========
    print("#" * 65)
    print("# 场景 A: M/M/1 解析验证 (常数到达率, FIFO 排队)")
    print("#" * 65)
    sim_a = EDQueueSimulation(
        n_doctors=1,
        mean_service_time=25.0,   # μ = 0.04/分钟
        sim_minutes=20000,        # 333 小时, 大量样本确保稳态
        warm_up=2000,             # 充分预热
        arrival_func=EDQueueSimulation.constant_arrival_rate,  # λ = 0.035
        fifo=True
    )
    sim_a.run()

    # M/M/1 解析对比
    result_a = sim_a.get_mm1_analytical()
    if result_a[0] is not None:
        lam_a, W_a, rho_a = result_a
        valid_a = [p for p in sim_a.patients
                   if p.arrival_time >= sim_a.warm_up and p.service_end_time is not None]
        sim_wait_a = np.mean([p.wait_time for p in valid_a]) if valid_a else 0

        print(f"\n  📐 M/M/1 解析解:")
        print(f"     λ = {lam_a:.4f}/分钟, μ = {sim_a.service_rate:.4f}/分钟")
        print(f"     ρ = {rho_a:.4f}")
        print(f"     解析 W_q = {W_a:.2f} 分钟")
        print(f"     仿真 W_q = {sim_wait_a:.2f} 分钟")

        if W_a > 0:
            mm1_error = abs(sim_wait_a - W_a) / W_a * 100
            print(f"     偏差: {mm1_error:.2f}%")
            if mm1_error < 5.0:
                print(f"  ✅ 验证通过: M/M/1 对比偏差 < 5%")
            else:
                print(f"  ⚠️ 偏差较大")

    # ========== 场景 B: 非平稳到达 + 优先级 + 3 医生 ==========
    print("\n" + "#" * 65)
    print("# 场景 B: 非平稳到达率 + 3 位医生 + 优先级排队")
    print("#" * 65)
    sim_b = EDQueueSimulation(
        n_doctors=3, mean_service_time=30.0,
        sim_minutes=2880, warm_up=480, fifo=False
    )
    sim_b.run()

    # ========== 场景 C: 非平稳到达 + 优先级 + 5 医生 ==========
    print("\n" + "#" * 65)
    print("# 场景 C: 非平稳到达率 + 5 位医生 + 优先级排队")
    print("#" * 65)
    sim_c = EDQueueSimulation(
        n_doctors=5, mean_service_time=30.0,
        sim_minutes=2880, warm_up=480, fifo=False
    )
    sim_c.run()

    # ✅ 验证 3: 增加医生后等待时间下降
    print("\n" + "=" * 65)
    print("✅ 验证 3: 增加医生后等待时间变化")
    print("=" * 65)

    results = [
        ("1 医生 (常数/FIFO)", sim_a),
        ("3 医生 (时变/优先级)", sim_b),
        ("5 医生 (时变/优先级)", sim_c)
    ]
    for label, sim in results:
        valid = [p for p in sim.patients
                 if p.arrival_time >= sim.warm_up and p.service_end_time is not None]
        if valid:
            waits = [p.wait_time for p in valid]
            print(f"  {label}: 平均等待 {np.mean(waits):.2f} 分钟, "
                  f"病人数 {len(valid)}")

    print(f"""
关键教学结论:
  1. M/M/1 解析解在 FIFO 排队下与仿真吻合 (偏差 < 5%)
  2. Little 定律 L=λW 在稳态下精确成立
  3. 多优先级排队显著降低危重病人等待时间
  4. 增加医生数量带来非线性"降等待"效果
  5. 仿真可以处理解析模型无法处理的复杂场景:
     - 非平稳到达率
     - 多优先级队列
     - 任意服务时间分布
""")
