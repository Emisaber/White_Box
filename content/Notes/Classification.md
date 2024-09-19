---
tags:
  - DL
  - ML
---

## How to do Classification

用宝可梦的种族值描述宝可梦作为输入  

### Classification as Regression

用数字表示类别，接近哪个数字代表哪个类别  
regression来做的话，隐含有数值大小的信息，输出是一个连续的数值范围，一方面会因为类别的离散化掩盖信息，一方面为了靠近离散的类别，训练结果可能出错  
![Pasted image 20240808114928](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240808114928.png)    
- 原本应是绿色分界线，为了使得右下角的数也趋近于1，得到紫色分界线

而且同时，因为数值的大小信息，表示类别的数字越接近，隐含着类别相接近的信息，往往是不成立的  

### 直接输出类别

![Pasted image 20240808115840](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240808115840.png)  
- model  ---  输入x，直接输出类别
- loss function ---  如果不一致则输出1，一致输出0，这样才能最小化。
- 无法微分，可以用preceptron, SVM  

### Classification as Probability

![Pasted image 20240809093845](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809093845.png)  
- 两个类别，在类别中取值的概率是$P(C)$，取值为$x$的概率是$P(x|C)$ ，给定$x$，属于某个类别的结果为$P(C|x)$  
- 计算概率，概率最大就可以得到所属类别
- 此时$P(C)$，$P(x|C)$ 需要从训练数据中计算出来

这样的想法称**Generative Model**  
通过计算$x$产生的概率，就能够产生$x$  
实际上在分类之前，产生$x$的概率就能被计算出来  

- 如何从训练数据中计算$P(C)$和$P(x|C)$
- $P(C)$直接由占比计算，$P(x|C)$必须能够代表在整个数据集中的概率
- **假设$C$是一个高斯分布**，需要从训练集中找到它的 $\mu$ 和 $\Sigma$ (Covariance matrix) 
	- ![Pasted image 20240809111421](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809111421.png)
- **Maximum Likelihood**

#### Maximum Likelihood
训练集中的数据，有可能从任何Gaussian Distribution中sample出来，但是得到这些数据的概率是不一样的  
找概率最大的Gaussian distribution ---Maximum Likelihood   

每个类别

$$Likelihood(\mu, \Sigma) = f_{\mu, \Sigma}(x^1) f_{\mu, \Sigma}(x^2) f_{\mu, \Sigma}(x^3)...$$  
$$\mu^*, \Sigma^* = argmax_{\mu,\Sigma}Likelihood(\mu,\Sigma)$$  
微分得解  
$$\mu^* = \frac{1}{N}\sum_{n=1}^Nx^n$$  
$$\Sigma^*=\frac{1}{N}\sum_{n=1}^N(x^n-\mu^*)(x^n-\mu^*)^T$$  
代入得到得$\mu$, $\Sigma$ 就可以得到对应的高斯分布，就做出来了。  

每个类别都找到Maximum likelihood的Gasussian ditribution，得到参数就可以计算概率  

在宝可梦属性分类任务表现糟糕  
![Pasted image 20240809110246](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809110246.png)
## Modifying Model

Generative model 中，更常见的是share the same $\Sigma$  
$\Sigma$跟input size平方成正比，如果每个都有独立的$\Sigma$ 的话，参数过多容易overfitting  
$\Sigma$共享的话  

两个类别的话  
$$Likelihood(u^1,u^2,\Sigma) = f_{u^1,\Sigma}(x^1)f_{u^2,\Sigma}(x^2)f_{u^3,\Sigma}(x^3)...$$  
$$\mu^i = \frac{1}{N^i}\sum_{n=1}^{N^i}x^n$$  
解得  
$$\Sigma^*=\frac{1}{N}\sum_{n=1}^N(x^n-\mu^*)(x^n-\mu^*)^T$$  
$$\Sigma=\frac{N_1}{M}\Sigma^1 + \frac{N_2}{M}\Sigma^2$$  

共享的情况下，分类的boundary会变成直线，称linear model  
![Pasted image 20240809110205](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809110205.png)  
正确率提高  

## Summary

- function set
	- ![Pasted image 20240809110617](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809110617.png)
- Loss function
	- maximum likelihood
- trainning(find the best function) --- 直接可以算  

##### Why Gaussian
自己定的，其他也可以，决定模型的复杂度  
如果是binary features(输入要么是0要么是1)，可以认为是Bernoulli distributions  
如果认为输入的每个维度是独立的，使用的是Naive Bayes Classifier(朴素贝叶斯分类器)  
![Pasted image 20240809112925](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809112925.png)  

##### Posterior Probability
![Pasted image 20240809113116](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809113116.png)  
- 上下同除$P(x|C_1)P(C_1)$，取自然对数，得到一个sigmoid
###### 关于z
![Pasted image 20240809113615](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809113615.png)  

![Pasted image 20240809113629](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809113629.png)  

![Pasted image 20240809113646](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809113646.png)  

相除相消  
![Pasted image 20240809113820](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809113820.png)

展开一通算  
![Pasted image 20240809114043](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809114043.png)  

$\Sigma$是共用的  
![Pasted image 20240809114219](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809114219.png)  

前一部分向量乘矩阵得到一个向量$w^T$，后一部分是一个常数$b$  
![Pasted image 20240809114400](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809114400.png)  

所以实际上  
![Pasted image 20240809114500](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240809114500.png)  
- 证明了当$\Sigma$共用时，Boundary是直线

在Generative model 中，估计了$N_1, N_2, \mu^1, \mu^2, \Sigma$ 来得到模型，那能不能直接找$\textbf{w}, b$呢？   
见[[Logistic Regression]]  


















