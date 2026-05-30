#!/usr/bin/env python3
"""
案例3：贝叶斯推断
===================
用贝叶斯定理更新对疾病概率的信念。
纯 Python 标准库实现，无第三方依赖。
"""
# 教学注释：围绕随机分布、样本统计量和蒙特卡洛估计观察不确定性。
# 运行时重点比较理论量与模拟结果如何支撑决策判断。


import math
import random


def bayes_update(prior, likelihood_positive, likelihood_negative):
    """
    贝叶斯更新：给定先验和似然，计算后验。

    参数：
        prior: P(D) 得病的先验概率
        likelihood_positive: P(T+|D) 灵敏度——病人检测阳性的概率
        likelihood_negative: P(T+|¬D) 假阳性率——健康人检测阳性的概率

    返回：
        posterior: P(D|T+) 检测阳性后得病的后验概率
        evidence: P(T+) 总阳性率（全概率公式）
    """
    evidence = likelihood_positive * prior + likelihood_negative * (1 - prior)
    if evidence == 0:
        return 0.0, 0.0
    posterior = likelihood_positive * prior / evidence
    return posterior, evidence


def bayesian_sequential_update(prior, likelihood_positive, likelihood_negative,
                                 num_positive_tests):
    """
    序贯贝叶斯更新：连续做 num_positive_tests 次阳性检测，
    每次把后验作为下一次的先验。
    """
    current_prior = prior
    posteriors = [current_prior]

    for i in range(num_positive_tests):
        posterior, _ = bayes_update(current_prior,
                                     likelihood_positive,
                                     likelihood_negative)
        current_prior = posterior
        posteriors.append(posterior)

    return posteriors


def bayesian_sequential_mixed(prior, likelihood_positive, likelihood_negative,
                                test_results):
    """
    序贯贝叶斯更新：支持混合的阳性/阴性检测结果。

    参数：
        test_results: 字符串列表，['+', '-', '+', ...]
    """
    current_prior = prior
    history = [(0, current_prior)]

    for i, result in enumerate(test_results):
        if result == '+':
            posterior, _ = bayes_update(current_prior,
                                         likelihood_positive,
                                         likelihood_negative)
        else:  # 阴性结果
            # 阴性似然：P(T-|D) = 1 - 灵敏度，P(T-|¬D) = 1 - 假阳性率
            posterior, _ = bayes_update(current_prior,
                                         1 - likelihood_positive,
                                         1 - likelihood_negative)
        current_prior = posterior
        history.append((i + 1, posterior))

    return history


def sensitivity_analysis(base_prior, likelihood_positive, likelihood_negative,
                          priors_to_test, num_tests=5):
    """
    敏感性分析：在不同的先验下，后验如何随阳性次数变化。
    """
    results = {}
    for prior in priors_to_test:
        posteriors = bayesian_sequential_update(prior,
                                                 likelihood_positive,
                                                 likelihood_negative,
                                                 num_tests)
        results[prior] = posteriors
    return results


def format_percent(p):
    """格式化百分比"""
    return f"{p * 100:.2f}%"


def main():
    print("=" * 60)
    print("案例3：贝叶斯推断——疾病检测")
    print("=" * 60)

    # 设置参数
    prevalence = 0.001        # 先验：发病率 0.1%
    sensitivity = 0.99        # 灵敏度：P(阳性 | 得病)
    false_positive = 0.01     # 假阳性率：P(阳性 | 健康)
    specificity = 1 - false_positive  # 特异度：P(阴性 | 健康)

    print(f"\n参数设置:")
    print(f"  发病率（先验）: {format_percent(prevalence)}")
    print(f"  检测灵敏度（真阳性率）: {format_percent(sensitivity)}")
    print(f"  假阳性率: {format_percent(false_positive)}")
    print(f"  检测特异度: {format_percent(specificity)}")
    print()

    # ========= 场景1：一次阳性 =========
    print("-" * 60)
    print("场景1：一次检测阳性")
    print("-" * 60)

    posterior, evidence = bayes_update(prevalence, sensitivity, false_positive)
    print(f"  总阳性率 P(阳性) = {format_percent(evidence)}")
    print(f"  P(得病 | 阳性) = {format_percent(posterior)}")
    print()
    print(f"  解释：检测准确率 99%，但得病概率只有 {format_percent(posterior)}")
    print(f"  远低于直觉的 99%。因为发病率极低 (0.1%)，")
    print(f"  大多数阳性结果是假阳性。")

    # ========= 场景2：连续两次阳性 =========
    print()
    print("-" * 60)
    print("场景2：连续两次检测阳性")
    print("-" * 60)

    posteriors = bayesian_sequential_update(prevalence, sensitivity,
                                             false_positive, 2)
    print(f"  第 0 次（先验）: {format_percent(posteriors[0])}")
    print(f"  第 1 次阳性后: {format_percent(posteriors[1])}")
    print(f"  第 2 次阳性后: {format_percent(posteriors[2])}")
    print()
    print(f"  两次阳性后，概率从 0.1% 升到约 {format_percent(posteriors[2])}，")
    print(f"  基本可以确诊。")

    # ========= 场景3：混合结果（先阳后阴） =========
    print()
    print("-" * 60)
    print("场景3：混合检测结果")
    print("-" * 60)

    results = ['+', '-', '+']
    history = bayesian_sequential_mixed(prevalence, sensitivity,
                                         false_positive, results)
    for step, prob in history:
        label = "先验" if step == 0 else f"第{step}次检测 ({results[step-1]})"
        print(f"  {label}: {format_percent(prob)}")

    # ========= 场景4：不同先验的敏感性分析 =========
    print()
    print("-" * 60)
    print("场景4：不同先验下的贝叶斯更新")
    print("（展示先验影响随证据增多而减弱）")
    print("-" * 60)

    priors = [0.0001, 0.001, 0.01, 0.1]
    sens_result = sensitivity_analysis(None, sensitivity, false_positive,
                                        priors, num_tests=5)

    header = f"{'阳性次数':>8}" + "".join(f"{format_percent(p):>12}" for p in priors)
    print(f"\n{header}")
    print(f"{'':->{len(header)}}")
    for i in range(6):  # 0 到 5 次
        row = f"{i:>8}"
        for prior in priors:
            row += f"{format_percent(sens_result[prior][i]):>12}"
        print(row)

    print()
    print("  解释：即使先验相差 1000 倍（0.01% vs 10%），")
    print("  5 次阳性后后验基本收敛到相同值。")
    print("  这说明：证据足够多时，先验的影响消失。")

    # ========= 场景5：直觉对比 =========
    print()
    print("-" * 60)
    print("场景5：直觉 vs 贝叶斯")
    print("-" * 60)

    print("""
大多数人直觉：
  "检测准确率 99%，阳性了 → 99% 概率得病"

贝叶斯的答案：
  P(病|阳) = 99% × 0.1% / 1.098% ≈ 9%

差距巨大的原因：
  1000 人中有 1 个得病（真阳性 ≈ 1 人）
  999 个健康人中有 ≈ 10 个假阳性
  总共 11 个阳性中只有 1 个是真阳性
  → 9% 是正确的答案！

下一次检测后：
  剩余的 990 个"健康人"中又产生约 10 个假阳性
  但这次先验是 9%（≈ 1 个真阳性 + 0 个假阳性）
  → 91% 是正确的答案！
    """)


if __name__ == "__main__":
    main()
