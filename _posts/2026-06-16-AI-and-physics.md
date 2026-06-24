---
title: AI 与物理学的关系思考
author: yuchong
date: 2026-06-16 00:34:00 +0800
categories: [AI]
tags: [AI，physics]
math: true
---

# 1. 玻尔兹曼分布与softmax

## 玻尔兹曼分布（统计力学）

在热力学平衡状态下，一个系统处于能量为$E_i$的微观状态的概率为：
$\begin{equation}
P(i) = \frac{e^{-E_i / k_B T}}{\sum_j e^{-E_j / k_B T}}
\end{equation}
$

其中$k_B$是玻尔兹曼常数，$T$是温度。

# 带退火温度的softmax

$\begin{equation}
\mathrm{softmax}_{T}(x_i) = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}}
\end{equation}
$
在数学形式上，和玻尔兹曼分布是一致的。

（1）当$T=1$: 就是标准的 Softmax；

（2）当$T→0^+$: 就是one-hot分布；在物理学上，系统逐渐“冻结”，越来越倾向于只接受能量更低的状态，最终收敛到全局或局部的能量最低点（基态）。

（3）当$T→∞$: 就是均匀分布；物理学角度来看，就是热运动剧烈，系统倾向于随机游走，接受差解的概率极高，从而跳出局部最优。

如果在算法上，执行（3）→（2），我们容易知道，这个就是著名的模拟退火算法（Simulated Annealing, SA），该算法用于在巨大搜索空间中寻找全局最优解的启发式概率优化算法。它的核心灵感来源于固体物理中的退火（Annealing）过程。
物理图像是：在冶金学中，如果要让一块金属的内部晶体结构达到最稳定、缺陷最少的状态（即能量最低的状态），工匠会这样做：

    （a）加热：把金属加热到极高的温度，此时内部粒子剧烈运动，处于完全无序的高能状态;

     (b) 缓慢冷却（退火）：让温度非常缓慢地下降。在这个过程中，粒子有足够的时间重新排列，找到更稳定的低能结构;

     (c) 冻结：当温度降到极低时，粒子被“冻结”在能量最低的晶格位置，金属变得极其坚固。


我们观察玻尔兹曼分布公式，容易知道：
$P_i \propto e^{-E_i / k_B T}$,能量项$e^{-E_i}$代表自然界有一种“惰性”的倾向，也就是处于最低能量状态（基态），能量越低，出现的概率越大。另外，第二项是温度项$k_BT$,代表了系统的运动剧烈程度，系统有熵增趋势；合并到一起就是：玻尔兹曼分布是大自然热平衡状态下在“追求最低能量”和“追求最大混乱度（熵）”之间达成的一份妥协协议，玻尔兹曼分布也是给定能量约束下熵最大的分布。

在AI领域，尤其是大模型时代，我们在训练的模型，其实都是在学习一种最佳的“无偏估计”，也就是“最大熵原理”：在满足给定约束条件下，选择熵最大的分布，相当于不添加任何多余的先验结构或偏见。

也就是热力学平衡态和AI的无偏估计是一回事，AI模型训练有一个loss函数，也就是对应的能量约束条件。

在大模型时代，最著名的还是Transformer中的注意力机制使用了softmax，我们在算法角度可以说是为了归一化分数，但是转到物理学角度，内涵就更加丰富了。
$
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
\end{equation}
$

另外，在大语言模型中，经过多层 Transformer 编码后，最后一个词的隐藏状态（Hidden State）会被送入一个巨大的线性层，比如为 N，这个线性层输出的 N 个原始分数被称为 Logits，接着会使用softmax映射，并使用Argmax选择最高概率的词。


# 2. 物理熵与信息熵

热力学第二定律：它有多种表述方式，核心思想是自然界的过程具有方向性。克劳修斯的表述是：热量不能自发地从低温物体转移到高温物体。开尔文勋爵的表述是：不可能从单一热源吸取热量，使之完全变为有用功而不产生其他影响。这一定律引入了熵（Entropy）的概念。熵是衡量一个系统“无序度”或“混乱程度”的物理量。热力学第二定律的本质是：在一个孤立系统中，总熵永远不会减少，只会增加或保持不变（在理想的、可逆的过程中）。简单来说，宇宙的趋势是从有序走向无序。热水会变凉，墨水滴入水中会扩散，这都是熵增的过程。

此时我们白日梦就是，如何逆转这个过程，也就是挑战热力学第二定律的权威。最先思考这个问题的人是著名理论物理学家麦克斯韦。

对于此问题，麦克斯韦构思了麦克斯韦妖的思想实验，后来解决这个思想悖论的核心是小妖的操作并非“免费”。为了做出正确的开关门决定，小妖必须获取关于分子速度的信息。物理学家利奥·西拉德（Leo Szilard）和莱昂·布里渊（Léon Brillouin）等人率先指出，信息与熵是紧密相关的。获取信息的过程本身就会影响物理系统。1948年，香农借用统计力学中玻尔兹曼的“熵”的概念，将热力学中的“系统混乱度”巧妙地转化为了信息论中的“信息的不确定性”或“信息量”。最终的解决方案由IBM的物理学家罗尔夫·兰道尔（Rolf Landauer）在1961年给出，即著名的兰道尔原理（Landauer's Principle）。兰道尔指出，信息的处理是有物理代价的，尤其是擦除信息这一操作。他认为：擦除1比特的信息，至少会向环境耗散 $k_B T \ln(2)$ 的能量，并产生等量的熵增。其中 T 是环境的绝对温度。这个最小能耗被称为“兰道尔极限”。


香农熵假设了一个拥有“无限算力”和“无限存储”的绝对观察者，当我们将视角切换到“有限算力”的观察者（人），并强调“可复用、可泛化的结构性信息”时，这就不再是一个纯粹的数学或通信问题，在经典热力学中，熵代表系统的无序度。而在有限算力的物理世界中，“提取可复用信息”本质上是一个“做功”的过程。在有限算力下，信息的提取不是免费的，在物理学中，有些问题（如多体量子系统的精确基态）在数学上是存在的，但在物理上是“不可计算”的（NP-Hard 或更糟）。如果观察者没有足够的算力，这些隐藏在数据中的结构就等同于“不存在”。在有限算力的约束下，信息等于系统内部能够被物理机制稳定维持、且能被观察者以有限能量（计算功）成功解码的有序结构。这篇文章《From Entropy to Epiplexity》是从香农熵到“认知复杂度”（Epiplexity）的跨越。

# 3. 在AIGC被复兴的VAE

VAE（Variational Autoencoder，变分自编码器）其实不是“新生物”，它最早出现在 2013–2014 年，由 Kingma 和 Welling 在一篇经典论文 “Auto-Encoding Variational Bayes” (2013, ICLR 2014 发表) 中提出。这篇论文首次系统化地把 概率图模型 与 深度学习的自编码器 结合起来，用 变分推断（variational inference）+ reparameterization trick 解决了采样不可导的问题，使得端到端的生成建模成为可能。VAE是图像生成模型的典型算法，其核心思想：将原始数据映射到一个已知分布，然后从已知分布中随机采样，通过生成模型得到与原始数据近似的数据。其本质上是对数据分布的拟合，然后从分布中采样得到新数据。

在生成模型中，我们假设可观测数据 x（如图像）是由某个不可见的隐变量 z（如特征、编码）生成的。根据贝叶斯定理，隐变量的后验概率分布为：

$
\begin{equation}
p(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|z)p(z)dz}
\end{equation}
$

分母需要对所有潜在特质z进行积分，在高维连续空间中，这个积分是不可积（Intractable）的，导致我们无法直接算出 $p(z\vert{}x)$。

变分推断的解法：既然算不出精确的 $p(z\vert{}x)$，我们就拿一个形式简单的已知分布 $q_{\phi}(z\vert{}x)$（通常假设为高斯分布）去主动拟合 $p(z\vert{}x)$。通过调整 $q$ 的参数 $\phi$，让它与真实后验的差距尽可能小。这个解法其实和VAE的核心思想是一致的。

为了衡量近似分布 $q_\phi(z\vert{}x)$ 与真实分布 $p(z\vert{}x)$ 的相似度，我们使用 KL 散度（Kullback-Leibler Divergence）：

$
\begin{equation}
D_{KL}(q_{\phi }(z|x)\parallel p(z|x))=\int q_{\phi }(z|x)\log \frac{q_{\phi }(z|x)}{p(z|x)}dz
\end{equation}
$

利用条件概率公式 $p(z\vert{}x) = \frac{p(x,z)}{p(x)}$ 将其展开变形：

$$
\begin{aligned}
D_{KL}(q_\phi(z|x) \parallel p(z|x)) &= \int q_\phi(z|x) \log \left( \frac{q_\phi(z|x) \cdot p(x)}{p(x, z)} \right) dz \\
&= \int q_\phi(z|x) \left[ \log \frac{q_\phi(z|x)}{p(x, z)} + \log p(x) \right] dz \\
&= \int q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(x, z)} \, dz + \log p(x) \int q_\phi(z|x) \, dz
\end{aligned}
$$

因为概率分布的积分 $\int q_\phi(z\vert{}x) dz = 1$，因此我们有：
$
\begin{equation}
\log p(x)=\int q_{\phi }(z|x)\log \frac{p(x,z)}{q_{\phi }(z|x)}dz+D_{KL}(q_{\phi }(z|x)\parallel p(z|x))
\end{equation}
$

又因为$p(x,z) = p_\theta(x\vert{}z)p(z)$,得到变分推断最著名的恒等式：

$$
\begin{equation}
\log p(x)=\operatorname{E}_{z\sim q_{\phi }(z\vert{}x)}[\log p_{\theta }(x\vert{}z)]-D_{KL}(q_{\phi }(z\vert{}x)\parallel p(z))+D_{KL}(q_{\phi }(z\vert{}x)\parallel p(z\vert{}x))
\end{equation}
$$
 
由于 KL 散度 $D_{KL}(q_\phi(z\vert{}x) \parallel p(z\vert{}x)) \geq 0$ 恒成立，因此公式前两项构成了 $\log p(x)$ 的下界，即 ELBO（Evidence Lower Bound，证据下界）:

$$
\begin{equation}
\text{ELBO}(\phi ,\theta ;x)=\operatorname{E}_{z\sim q_{\phi }(z\vert{}x)}[\log p_{\theta }(x\vert{}z)]-D_{KL}(q_{\phi }(z\vert{}x)\parallel p(z))
\end{equation}
$$
由此可以得到：

$$
\begin{equation}
\log p(x)\ge \text{ELBO}(\phi ,\theta ;x)
\end{equation}
$$
由此可知，在训练 VAE 时，我们的目标是最大化边际似然 $\log p(x)$。由于真实后验不可知，我们无法直接最小化第三项的 KL 散度。但通过公式可以发现：最大化 ELBO 相当于在同时“最大化数据似然”并“最小化近似后验与真实后验的差距”,这个也是变分推断的精妙之处。

那么这两项的含义是什么呢？

第一项，期望在近似后验分布 $\log p_{\theta}(x\vert{}z)$ 下采样出的 $z$，能够被解码器很好地重建出原数据 $x$，它鼓励重建的图像和原图越像越好。第二项，鼓励编码器输出的分布 $q_{\phi }(z\vert{}x)$ 不要偏离我们预先设定的先验分布 $p(z)$。

在物理学中，自由能（Free Energy）的核心含义可以通俗地理解为：在一个热力学系统中，在特定条件下（如恒温恒压或恒温恒容）真正“可用”来做有用功的那部分能量。
它之所以被称为“自由”，是因为这部分能量不受系统内部无序度（熵）的“束缚”，可以被系统自由地释放出来对外做功。我们会使用吉布斯自由能G和亥姆霍兹自由能F。
亥姆霍兹自由能 $F$ 的定义为：$F=E-TS$,

E是系统的内能（系统倾向于寻找能量最低、最稳定的状态，比如球往低处滚）。T 是温度，S 是熵（代表系统的混乱度，热力学第二定律决定了系统倾向于越混乱越好，即熵增）。物理趋向：一个自发的物理系统，总是在低内能（最小化 E）和高混乱度（最大化 S）之间寻找平衡，也就是使得自由能 F 达到最小。这个就是和玻尔兹曼分布是一样的，自由能重点关注系统的演化过程倾向，也就是寻找平衡态的过程，而玻尔兹曼分布主要说最终的平衡态的特征。

对照ELBO的优化目标：

| vae优化项 |  统计物理  | 说明 |
| :---: | :---: | :---: |
| ELBO | 自由能| 系统自发走向最稳定、最和谐的状态，确定性与随机的权衡 |
| 第一项重建优化 |内能E| 希望隐变量 z 和数据 x 高度匹配，能量越低，数据解释得越准 |
| 趋于先验分布 | $-TS$| 强制要求隐空间 p(z) 保持一定的混乱度（熵），防止其坍塌成一个死板的孤立点。 |



# 4. Hopfield 网络

约翰·J·霍普菲尔德（John J. Hopfield）因“通过人工神经网络实现机器学习的基础性发现和发明”，与杰弗里·E·辛顿（Geoffrey E. Hinton）共同获得了2024年诺贝尔物理学奖。霍普菲尔德利用描述磁性材料中自旋相互影响的物理学原理，建立了一个具有节点和连接的模型网络。他将神经网络建模成一个具有“能量函数”的物理系统，网络在运行中会寻找能量最低的稳定点，从而实现了信息的存储与回忆。

![图1](assets/image/20260621/1.png)

这篇John J. Hopfield于1982年发表的文章《Neural networks and physical systems with emergent collective computational abilities》,从文章题目就看到了物理的来源,其网络结构如下：

![图2](assets/image/20260621/2.png)

Hopfield神经网络是一种单层互相全连接的反馈型神经网络。每个神经元既是输入也是输出，网络中的每一个神经元都将自己的输出通过连接权传送给所有其它神经元，同时又都接收所有其它神经元传递过来的信息。即：网络中的神经元在t时刻的输出状态实际上间接地与自己t-1时刻的输出状态有关。神经元之间互连接，所以得到的权重矩阵将是对称矩阵。

和多层神经网络一样，Hopfield神经网络也有其训练目标函数，所使用的是能量函数，网络在运行过程中会通过状态更新逐步降低该能量值，最终收敛到一个局部极小点，这个点对应一个稳定的记忆模式或优化解，具体形式是：
$
\begin{equation}
E = -\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij} x_i x_j + \sum_{i=1}^{n} b_i x_i
\end{equation}
$

我们一眼就可以看到，这个其实是统计物理中最经典的模型-伊辛模型（Ising Model）的哈密顿量（Hamiltonian），其标准形式是：
$
\begin{equation}
H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i
\end{equation}
$

$H$：系统的哈密顿量（Hamiltonian），代表系统的总能量。

$J$：交换耦合常数，决定相邻自旋之间相互作用的强度与方向（$J > 0 $为铁磁耦合，$J < 0 $为反铁磁耦合）。

$⟨i,j⟩$：表示对最近邻自旋对求和，即只对空间中相邻的两个格点 $i$ 和 $j$ 进行配对计算。

$sᵢ, sⱼ$：第 $i $ 个和第 $j$ 个格点上的自旋变量，通常取值为 +1 或 -1。

$h$：外加磁场强度，影响每个自旋的取向。

$∑ᵢ sᵢ$：对所有格点上的自旋求和，表示外场对系统总能量的贡献。

这种类比使得Hopfield网络能够模拟“联想记忆”——当输入一个不完整或带噪声的模式时，网络会通过状态更新逐步降低能量，最终收敛到一个稳定的“吸引子”状态，即最接近的存储记忆。这个过程与物理系统中自旋趋向能量最低态的过程完全一致。

# 5. 玻尔兹曼机（Boltzmann Machine）

玻尔兹曼机（Boltzmann Machine, BM）是一种基于概率图模型的神经网络，它以物理中的玻尔兹曼分布为基础，将能量函数与概率分布联系起来，是一种无监督学习模型，玻尔兹曼机使用能量函数来描述系统的状态。网络状态的概率由其能量决定，根据统计物理原理，能量越低的状态出现的概率越高。玻尔兹曼机通过模拟退火的方法调整系统状态，寻找最小能量的配置。其训练目标是最大化输入数据的似然函数，学习数据分布。

受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）：RBM是一种特殊的玻尔兹曼机，简化了网络结构，可见层和隐藏层之间是全连接的，但同一层内没有连接。这种结构便于训练。

深度玻尔兹曼机（Deep Boltzmann Machine, DBM）：由多个RBM堆叠而成，用于学习更深层次的特征。

玻尔兹曼机虽已不是当前深度学习的主流模型，但其作为连接统计物理与机器学习的桥梁，为深度学习的发展提供了关键思路，如无监督预训练、生成式建模等。量子玻尔兹曼机有望解决经典玻尔兹曼机的训练瓶颈，在药物研发、金融优化等领域发挥更大作用。


### 与Transformer的内在结构联系
Transformer是一个“动态权重版”的伊辛模型, attention = 动态相互作用 $ W_{ij} $, token = 自旋.

![图3](assets/image/20260621/3.png)
《Hopfield Networks is All You Need》这篇文章中，提到：

提出了一种具有连续状态和相应更新规则的现代霍普菲尔德网络。这种新型霍普菲尔德网络能够以指数级（相对于联想空间维度）存储大量模式，新的更新规则等价于Transformer中使用的注意力机制。这一等价性使得我们可以对Transformer模型中的注意力头进行刻画：在前几层，这些注意力头主要执行全局平均操作；而在深层，则通过亚稳态实现局部平均。这种新型现代霍普菲尔德网络可作为层嵌入深度学习架构中，用于存储并访问原始输入数据、中间计算结果或已学习的原型。这些霍普菲尔德层为深度学习提供了超越全连接、卷积或循环网络的新路径，并实现了池化、记忆、联想和注意力机制.


# 6. 扩散模型


扩散模型的灵感源于非平衡热力学。2015年，斯坦福大学的 Jascha Sohl-Dickstein 等人在论文《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》中首次引入了扩散模型的概念。该研究将物理扩散过程（如一滴墨水在水中扩散）引入机器学习，提出了通过前向过程逐步向数据添加噪声，再通过反向过程学习去噪来生成数据的基本思想。

Yang Song 和 Stefano Ermon 提出了基于分数的生成模型，专注于学习数据分布的梯度，对现代扩散模型的发展至关重要。

Jonathan Ho 等人发表了《Denoising Diffusion Probabilistic Models》（DDPM），通过简化训练目标并采用 U-Net 架构，系统性地提出了去噪扩散概率模型。DDPM 展示了扩散模型能够生成高质量图像，标志着其开始与当时占主导地位的生成对抗网络（GANs）竞争。

DDPM的核心是学习数据的分布：$x_0 \sim q(x_0)$,目标是构建一个模型$p_\theta (x_0)$,使其逼近真实数据分布。我们的优化目标是：$\max_{\theta} \log p_{\theta}(x_0)$, 其中:

$$p_{\theta}(x_0) = \int p_{\theta}(x_{0:T}) \, \mathrm{d}x_{1:T}$$

也就是把我们把所有“隐藏变量 $x_{1:T}$”都积分掉，只留下 $x_0$ 的概率.
因此有：


$$
\begin{aligned}
\log p_{\theta}(x_0) &= \log \int p_{\theta}(x_{0:T}) dx_{1:T}\\
&=\log \int q(x_{1:T} | x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} dx_{1:T}\\

&=\log \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]
\end{aligned}
$$

根据：$\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$

$$
\begin{equation}
\log p_\theta(x_0) \geq \mathbb{E}_q \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]
\end{equation}
$$

我们知道：
$
\begin{equation}
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)
\end{equation}
$

$$
\begin{equation}
q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})
\end{equation}
$$

得到（12）式右侧变为：

$$
\begin{equation}
\begin{aligned}

\mathbb{E}_q \left[ \log p(x_T) + \sum_{t=1}^{T} \log p_\theta(x_{t-1}|x_t) - \sum_{t=1}^{T} \log q(x_t|x_{t-1}) \right]\\
= \mathbb{E}_q[\log p(x_T)] + \sum_{t=1}^{T} \mathbb{E}_q [\log p_\theta(x_{t-1}|x_t) - \log q(x_t|x_{t-1})]

\end{aligned}
\end{equation}
$$

根据概率链式法则（先从$x_0$到$x_{t-1}$,再从$x_{t-1}$到$x_{t}$）：

$$q(x_{t-1}, x_t | x_0) = q(x_{t-1} | x_0) \, q(x_t | x_{t-1})$$

逆向反推链式关系：

$$q(x_{t-1}, x_t | x_0) = q(x_t | x_0) \, q(x_{t-1} | x_t, x_0)$$

因为它们是同一个概率分布，所以：
$$q(x_{t-1}|x_0) \, q(x_t|x_{t-1}) = q(x_t|x_0) \, q(x_{t-1}|x_t, x_0)$$


$$q(x_t|x_{t-1}) = \frac{q(x_t|x_0) \, q(x_{t-1}|x_t, x_0)}{q(x_{t-1}|x_0)}$$

$$\log q(x_t|x_{t-1}) = \log q(x_{t-1}|x_t, x_0) + \log q(x_t|x_0) - \log q(x_{t-1}|x_0)$$

替换到（15）式中，
$$
\begin{aligned}
-\sum_{t=1}^{T} \log q(x_t|x_{t-1})&=-\sum_{t=1}^{T}(\log q(x_{t-1}|x_t, x_0) + \log q(x_t|x_0) - \log q(x_{t-1}|x_0))\\
&=\sum_{t=1}^{T}(-\log q(x_t|x_0)+\log q(x_{t-1}|x_0))-\sum_{t=1}^{T}(\log q(x_{t-1}|x_t, x_0))\\
&=\log q(x_0|x_0)-\log q(x_T|x_0)-\sum_{t=1}^{T}(\log q(x_{t-1}|x_t, x_0))\\
&=\log q(x_T|x_0)-\sum_{t=1}^{T}(\log q(x_{t-1}|x_t, x_0))\\
&=-\log q(x_T|x_0)-\sum_{t=1}^{T}(\log q(x_{t-1}|x_t, x_0))
\end{aligned}
$$


最终（12）式变为：

$$
\begin{equation}
\log p_\theta(x_0) \geq \mathbb{E}_q \left[ \log p(x_T) - \log q(x_T|x_0) + \sum_{t=1}^T (\log p_\theta(x_{t-1}|x_t) - \log q(x_{t-1}|x_t, x_0)) \right]
\end{equation}
$$

我们知道：

$$
KL(q \| p) = \mathbb{E}_q[\log q - \log p]
$$

针对（16）式，我们有：

$$
\mathbb{E}_q[\log p(x_T) - \log q(x_T|x_0)] = -KL(q(x_T|x_0) \| p(x_T))
$$
$$
\mathbb{E}_q[\log p_\theta(x_{t-1}|x_t) - \log q(x_{t-1}|x_t, x_0)] = -KL(q \| p_\theta)
$$

当$t=1$时，$q(x_0 \vert{}x_1, x_0) = \delta(x_0)$,代表的含义是：在已经知道 $x_0$ 和 $x_1$的条件下，$x_0$的分布是什么？也就是不存在随机性，与模型的优化参数无关。最终（16）式变为：

$$
\begin{equation}
\log p_\theta(x_0) \geq \mathbb{E}_q[\log p_\theta(x_0|x_1)] - \sum_{t=2}^T KL(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) - KL(q(x_T|x_0) \| p(x_T))
\end{equation}
$$
这3项分布代表：起点重构项，中间逐步匹配KL项，和终点KL项。

一个概率图模型，全噪声的概率比较低，真实世界的所有高清图片，都分布在极其高维空间中的某些特定区域（我们称之为“数据流形”）。在这个高维空间里，每一点都有一个“概率值”，扩散模型学习的目标就是从任何地方出发，都指向高概率的区域，因此概率梯度变化最快的路径，就是模型需要学习的路径。

我们知道：

$$
p(x_t) = \int p(x_t, x_0) \, dx_0=\int p(x_t|x_0)p(x_0) \, dx_0\\
$$

$$
\begin{equation}
\nabla \log f = \frac{\nabla f}{f}
\end{equation}
$$
所以：
$$
\nabla_{x_t} \log p(x_t) = \frac{\nabla_{x_t} p(x_t)}{p(x_t)}
$$

对$x_t$求导：

$$\nabla_{x_t} p(x_t) = \int \nabla_{x_t} p(x_t | x_0) p(x_0) \, dx_0$$

根据（18）式得到：$$\nabla p(x_t | x_0) = p(x_t | x_0) \nabla \log p(x_t | x_0)$$ 
代入上式：

$$
\nabla p(x_t) = \int p(x_t | x_0) p(x_0) \nabla \log p(x_t | x_0) \, dx_0
$$

又因为：$p(x_t \vert{}x_0) p(x_0)=p(x_t,x_0)$，

所以：
$$\nabla p(x_t) = \int p(x_t, x_0) \nabla \log p(x_t | x_0) \, dx_0$$

代回 log 梯度:

$$
\nabla \log p(x_t) = \frac{1}{p(x_t)} \int p(x_t, x_0) \nabla \log p(x_t | x_0) \, dx_0
$$

根据Bayes公式有：$p(x_0 \vert{}x_t) = \frac{p(x_t, x_0)}{p(x_t)}$,代入上式：

$$\nabla \log p(x_t) = \int p(x_0 | x_t) \nabla \log p(x_t | x_0) \, dx_0$$

得到：
$$score= \nabla_{x_t} \log q(x_t) = \mathbb{E}_{q(x_0 | x_t)} [\nabla_{x_t} \log q(x_t | x_0)]$$

对于高斯分布下的KL散度，经过进一步简单推导就能知道以下几个对象，都有明确的转换关系：

$$KL \iff \|\mu_q - \mu_\theta\|^2 \iff \|\epsilon - \epsilon_\theta\|^2 \iff \|\text{score}_q - \text{score}_\theta\|^2$$

训练的目标=在不同噪声强度下，学习去噪方向（score）,且这个score等价于预测噪声：
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

如果在物理角度，其实是学习一个时间条件 score field：
$$s_\theta(x, t) \approx \nabla_x \log p_t(x)=-\frac{\epsilon}{\sigma_t}$$

### 物理图景

（1）概率梯度（Score）之所以是学习目标，是因为只要掌握了梯度场，生成样本就退化成了一个简单的“顺流而下”的积分过程。这就是为什么扩散模型能生成极其高质量、多样化样本的根本原因。也就是学习了一个类似的低概率到高概律的物理场，生成过程只要在这个物理场中，不用管最终目标，结果必然会流向高概率，这个高概率也必然就是我们的目标。

（2）在统计力学中，概率和能量是绑定的，玻尔兹曼分布：概率正比于 $e^{-E}$
   - 高概率区域 = 低能量区域（也就是物理上的“稳定态”或“势能谷底”）。
   - 低概率区域 = 高能量区域（物理上的“不稳定态”）。

  扩散模型学习的 Score 场，本质上就是势能场的负梯度（也就是受力方向）。

（3）我们曾经痴迷于“实体”——试图让 AI 记住每一张猫的照片，或者记住“猫”这个概念。
但扩散模型的出现，标志着 AI 开始用“场”的思维来理解世界。
   - Score 场：告诉你概率的梯度方向。
   - 能量场：告诉你当前状态有多“稳定”。
   - 速度场：告诉你如何从噪声平滑地流向数据（这就是目前大火的 Flow Matching 理论）。

## 7. Scaling law 


### Scaling Law 与临界标度律的形式同构

机器学习中的 Scaling Law（如 loss 随模型规模、数据量、算力的幂律下降）呈现出一个非常稳定的经验结构：

$$L(N) \sim N^{-\alpha}, \quad L(D) \sim D^{-\beta}, \quad L(C) \sim C^{-\gamma}$$

幂律关系在多个数量级上成立,指数 α,β,γ 对架构变化不敏感, 这一点在 Kaplan et al.（2020）中被系统化描述，并形成一个经验事实:“Changing architecture does not significantly change the scaling exponent.”

在统计物理中，临界点附近的系统也呈现同构结构(临界标度律（critical scaling law）)：
$$\xi \sim |T - T_c|^{-\nu}, \quad M \sim |T - T_c|^{\beta}$$

同样表现出：a.幂律（power law）,b.指数跨材料不变,c.微观细节不影响临界指数,d.在多个尺度上成立.

| 物理系统               | AI系统                               |
| ------------------ | ---------------------------------- |
| 微观粒子相互作用           | 神经网络参数更新机制                         |
| RG coarse-graining | 深层表示压缩 / feature abstraction       |
| universality class | 模型族（Transformer） |
| critical exponent  | scaling exponent (α, β, γ)         |


### 统计物理临界标度律（critical scaling law）机制来源：重整化群（RG）作为潜在生成机制


在物理中，临界标度律的深层机制是重整化群（Renormalization Group）不动点，其核心思想是：
- 系统不断 coarse-graining（尺度变换）
- 参数流向 RG flow
- 在临界点附近收敛到不动点
- 不动点决定所有临界指数

### AI 领域的Scaling Law机制是什么？

AI 领域的Scaling Law其机制层面的等价性仍未建立，猜测：AI可能处在某种高维统计系统的“类临界区间”，而非严格意义上的RG不动点系统。

## 8. 物理与AI的其他关联展望
还有很多其他的AI与物理学相似的地方，例如：涌现与相变，物理中最核心的对称性理论（物理中除了场的思想之外，最具深刻的思想）与AI大模型的架构联系，甚至与人类大脑的结构关系等等，都是非常有趣的现象，值得思考。