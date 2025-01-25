---
tags:
  - ML
  - DL
  - LLM
---


Attention mechanism in Transformer can be expressed as:   

$$
\mathrm{A t t e n t i o n} ( Q, K, V )=\mathrm{s o f t m a x} \left( \frac{Q K^{T}} {\sqrt{d_{k}}} \right) V 
$$

The purpose of dividing by $\sqrt{d_k}$  

- Control the magnitude of Scores 控制注意力分数的大小
	- 注意力的大小通过向量(key vector)点乘计算，向量维度变大，注意力分数可能随着变大，过大的注意力分数可能导致梯度爆炸等问题。通过除以$\sqrt{d_k}$来控制注意力分数的大小  
- Maintain Distribution Stability  维持分布的稳定性
	- 当queries, keys 服从高斯分布时(特别是初始化的时候)，他们的点乘会带来$d_k$的方差，除以$\sqrt{d_k}$ 将方差归一化到1。当方差过大时，softmax会只关注数值大的输入，产生很小的梯度，阻碍模型学习
- Heuristic improvement
	- 除以 $\sqrt{d_k}$ 应该也是实验的结果
