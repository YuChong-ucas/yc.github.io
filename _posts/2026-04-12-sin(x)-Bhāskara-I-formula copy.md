---
title: sin(x)的Bhāskara I估计公式修正
author: yuchong
date: 2026-04-12 00:34:00 +0800
categories: [mathematic]
tags: [mathematic]
math: true
---

# Bhāskara I 公式

Bhāskara I公式是古印度数学家 Bhāskara I 在公元7世纪提出的一个用于近似计算正弦函数sin(x)的经典公式，是历史上最早、最优雅的三角函数近似之一。
原本的Bhāskara I估计公式为（$[0,π]$区间上）：

$\sin(x) \approx \frac{16x(\pi - x)}{5\pi^2 - 4x(\pi - x)}$

# 优化新版本（update_Bhāskara I）

对Bhāskara I公式进行优化，得到新的估计公式（个人算法优化实践中的总结，不一定正确，欢迎大家指正）：

$\sin(x) \approx \frac{3305x(\pi - x)}{1050\pi^2 - 1018x(\pi - x) + 50\bigl[x(\pi - x)\bigr]^2}$

验证精度：

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 定义原 Bhaskara I 公式
# sin(x) ≈ 16 x (π-x) / (5 π² - 4 x(π-x))
# -------------------------------
def sin_bhaskara(x):
    X = x * (np.pi - x)
    return 16 * X / (5 * np.pi**2 - 4 * X)

# -------------------------------
# update_Bhaskara I
# sin(x) ≈ 3305 X / (1050 π² - 1018 X + 50 X²) ；其中X=x(π-x)
# -------------------------------
def sin_bhaskara_upgraded(x):
    X = x * (np.pi - x)
    return 3305 * X / (1050 * np.pi**2 - 1018 * X + 50 * X**2)

# -------------------------------
# 生成数据
# -------------------------------
x_data = np.linspace(0, np.pi, 1000)
y_true = np.sin(x_data)

y_bha = sin_bhaskara(x_data)
y_upg = sin_bhaskara_upgraded(x_data)

# -------------------------------
# 可视化对比
# -------------------------------
plt.figure(figsize=(12,5))

# 曲线对比
plt.subplot(1,2,1)
plt.plot(x_data, y_true, 'k', label='sin(x)')
plt.plot(x_data, y_bha, '--', label='Bhaskara I')
plt.plot(x_data, y_upg, '-.', label='update_Bhaskara I')
plt.title("sin(x) vs Bhaskara I & update_Bhaskara I")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 误差对比
plt.subplot(1,2,2)
plt.plot(x_data, np.abs(y_true - y_bha), '--', label='Bhaskara I error')
plt.plot(x_data, np.abs(y_true - y_upg), '-.', label='update_Bhaskara I error')
plt.title("Absolute error comparison")
plt.xlabel("x")
plt.ylabel("|formula - sin(x)|")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 最大误差
max_error_bha = np.max(np.abs(y_true - y_bha))
max_error_upg = np.max(np.abs(y_true - y_upg))

print("Bhaskara I 最大误差:", max_error_bha)
print("update_Bhaskara I最大误差:", max_error_upg)



```

![图1](assets/image/20260412/1.png)

        Bhaskara I 最大误差: 0.001631763696868127
        update_Bhaskara I最大误差: 0.00012133726370736064






