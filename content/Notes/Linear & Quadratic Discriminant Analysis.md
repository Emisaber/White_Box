---
tags:
  - DL
  - ML
---

>LDA & QDA  
>线性判别分析 &  二次判别分析  
## Why LDA ？

- stable
- preferred when multipel classes

##### 逻辑斯谛在区分度高时不稳定
逻辑斯谛回归(logistic regression) 当类别区分度高时，特征可以很好的区分类别，此时逻辑斯谛回归会出现完全分离(Complete Separation)或准完全分离(Quasi-complete Separation)，此时决策边界完全地区分了类别。

此时，要求对于正类有$p_i \approx 1$  ，对于负类有$p_i \approx 0$   
因为概率$p_i$为  
$$
p_i= \sigma(X_i) = \frac{1}{1 + e^{-X_i\beta}}
$$  
为了达到1，$\beta$ 需要尽可能大  
同时，逻辑斯谛的目标函数是最大化对数似然函数  
$$
J(\beta) = \sum_{i=1}^n[y_ilog(p_i) + (1-y_i)log(1-p_i)]
$$  
对数似然函数对$\beta$ 没有限制，$\beta$ 越大，目标函数越大，无法收敛  

##### 计算量
当类别大于两类时，逻辑斯谛计算量较大，不如LDA简单  

## From course
### LDA

LDA 来自于 贝叶斯定理  
$$
P(Y=k|X=x) = \frac{P(X=x|Y=k)P(Y=k)}{P(X=x)}
$$  
已知先验概率$P(Y=k)$，以此求解后验概率 $P(Y=k|X=x)$   
将贝叶斯公式改写为  
$$
P(Y=k|X=x) = \frac{\pi_kf_k(x)}{\sum_{l=1}^K\pi_lf_l(x)}
$$  
- $\pi_k$ 表示 $P(Y = k)$ 即先验概率
- $f_k(x)$ 表示 $P(X=x|Y=k)$，是第k类观测的X的密度函数

#### 单个类别

假设 $f_k(x)$ 是高斯分布，有(LDA适用于其他分布)   
$$
f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k}e^{-\frac{1}{2}(\frac{x-\mu_k}{\sigma_k})^2}
$$  
又假设每个类别的方差$\sigma_k$相等，记作$\sigma^2$   

将$f_k(x)$代入可得  

$$
p_{k} ( x )=\frac{\pi_{k} \frac{1} {\sqrt{2 \pi} \sigma} e^{-\frac{1} {2} \left( \frac{x-\mu_{k}} {\sigma} \right)^{2}}} {\sum_{l=1}^{K} \pi_{l} \frac{1} {\sqrt{2 \pi} \sigma} e^{-\frac{1} {2} \left( \frac{x-\mu_{l}} {\sigma} \right)^{2}}} 
$$  
- 上下的$\frac{1}{\sqrt{2\pi}\sigma}$ 是一样的，约去 
- 拆开e的指数，$\frac{x^2}{\sigma^2}$ 是一样的，约去  
- 约去之后，下面是一个常数(和固定不变)，分类结果($p_k(x)$) 取决于分子  

即，贝叶斯分类器将观测分类到  
$$
\delta_k({x}) = x\frac{\mu_k}{\sigma^2} - \frac{\mu ^2}{2\sigma^2} + log(\pi_k)
$$  
可以看到，此时的$\delta$是线性的  

假设K = 2, $\pi_1 = \pi_2 = 0.5$，则贝叶斯决策边界(概率相同的边界)(令两个$\delta$相等)  
$$
X = \frac{\mu_1 + \mu_2}{2}
$$  

> 贝叶斯决策边界类别概率相等，产生最少的错误分类  

#### 多预测变量

假设X 服从均值不同，协方差矩阵相同的多元高斯分布。  
>多元高斯分布假设每个预测变量服从正态分布，而且预测变量之间存在相关性

多元高斯分布密度  
$$
f ( x )=\frac{1} {( 2 \pi)^{p / 2} | \Sigma|^{\frac{1} {2}}} e^{-\frac{1} {2} ( x-\mu)^{T} \Sigma^{-1} ( x-\mu)} 
$$  

由于假设的协方差矩阵相同，二次项同样被约去，得到判别函数   
$$
\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + log(\pi_k)
$$


仍然线性   

### QDA
二次判别分析    
Quadratic Discriminant Analysis  

假设每一类更观测的$f_k(x)$都服从一个多元高斯分布，且每一类观测都有自己的协方差矩阵。  

由于协方差不再能约去，二次判别分析的决策边界不再是线性的   
$$
\delta_k(x) = -\frac{1}{2}(x - \mu_k)^T\Sigma_k^{-1}(x-\mu_k) + log\pi_k - \frac{1}{2}log|\Sigma_k|
$$   

### Beyond Course

#### assumptions

- Both LDA and QDA assume the predictor variables X are drawn from multivariate Gaussian distribution  
- LDA assumes equality of covariances among the predictor variables X across all class y.  (which relaxed by QDA)
- LDA and QDA require the number of predictor variables $p$ to be less than the sample size $n$, (works well when $n \ge 5 \times p$)   

#### Comparing Logistic and DIscriminant Analysis

- When assumptions of Discriminant Analysis happened, LDA&QDA would have better preformance, otherwise logistic may outperforms them
- both of LDA and logistic produce linear decision boundary, so that if decision boundary is non-linear, QDA would be the best
- 


## Reference
- [Linear & Quadratic Discriminant Analysis · UC Business Analytics R Programming Guide](https://uc-r.github.io/discriminant_analysis)   
- [ISLR](https://www.statlearning.com/)

