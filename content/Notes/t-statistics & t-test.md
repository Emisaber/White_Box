---
tags:
  - Math
  - statistic
---

# t-statistics

### 粗略的看

从样本中sample出一堆$X_i$，得到$X_i$的均值$\bar X$    
在统计中，认为$\bar X$服从正态分布，即  
$$\bar X\sim N(\mu_{\bar X}, \sigma_{\bar X}^2)$$  
为了知道当前的$\bar X$处于分布的哪个位置(即计算当前得到的样本均值$\bar X$的概率)，需要计算$\bar X$与均值之前差了多少个$\sigma$，即  
$$\frac{\bar X-\mu_{\bar X}}{\sigma_{\bar X}}$$  

但是实际上$\sigma_{\bar X}$难以获得，认为$$\sigma_{\bar X} = \frac{\sigma}{\sqrt{n}}$$   

>由IID假设同样能得到上式，此处的$\sigma$是样本总体的标准差

故与$\mu$的距离为  
$$\frac{\bar X-\mu}{\frac{\sigma}{\sqrt{n}}}$$   
被称为**Z-statistic or Z-score**  
$$Z = \frac{\bar X-\mu}{\frac{\sigma}{\sqrt{n}}}$$  
但是，我们也不知道$\sigma$ 是多少，通过样本($X$)的标准差来近似，即原式改为  
$$Z = \frac{\bar X-\mu}{\frac{S}{\sqrt{n}}}$$   
当样本数大于等于30时，这种近似是可行的   
同时，当样本总数大于等于30时，Z可以被看成是正态分布   

此时，如果样本数小于30，分布变为t分布   
即**t-statistic**   
$$t = \frac{\bar X-\mu}{\frac{S}{\sqrt{n}}}$$  

无论是哪种分布/统计量，通过查表能够得到取得当前$\bar X$的概率   


### 更进一步的

> _In statistics, the t-statistic is the ratio of the difference in a number’s estimated value from its assumed value to its standard error. It is used in hypothesis testing via Student's t-test._   

**t统计量是估计值与假设值之差，与其标准误差之间的比率**    
在t-test(t检验)中，用于计算P值  

它与Z-score(Z分数)十分相似，当样本量较小，或者这总体标准差未知时，使用t统计量  

公式为  
$$t_{\hat \beta} = \frac{\hat \beta - \beta_0}{SE(\hat \beta)}$$   

其中，$\hat \beta$ 是估计量，$\beta_0$ 是假设值，$SE(\hat \beta)$ 是[[Standard Error 标准误差]]   

在经典线性回归模型中，如果$\hat \beta$是普通最小二乘估计量(正态分布和同源性)，并且$\beta$的真实值是$\beta_0$，在t统计量的**抽样分布**是自由度$n-k$ 的**t分布**，其中k是系数个数(回归变量的数量)(包括截距)  

大多数模型，只要满足以下条件，t统计量就渐近具有标准正态分布(t分布)  
- $\hat \beta$ 与 $\beta$是一致的，并且是正态渐近分布
- 参数$\beta$的真实值是$\beta_0$
- $SE$正确估计了样本的标准差

实际上，t统计量是一个估计值(抽样计算得到的)和假设值(认为的真实值)之间距离的衡量，由于不能单纯使用做差描述距离(没有考虑抽样样本分布的信息，这样定义的远近并不合适)，引入标准差描述抽样样本对真实值的接近程度来定义距离   

距离越远，t统计量越大，二者差异越大  


# t-test

>_Student's t-test is a statistical test used to test whether the difference between the response of two groups is statistically significant or not. It is any statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis._

学生t检验，用于检验两组样本的统计量之间的差异是否具有统计学意义   
只要在0假设下满足t分布，即可使用t检验   

通常t检验的结果和Z检验相似，t检验随着样本数量增加会收敛到Z检验   

经常用于计算两组均值之间的差异是否显著   


### Assumptions

- IID 独立同分布进行随机取样
- 检验的两组样本需要是近似于正态分布的
- 两组样本的方差得是相同的
- 没有离群点

### Degree of freedom

$$df = \sum n_s - 1$$   
t检验的自由度是样本数减一  


### Uses of t-test

- One sample t-test test 单样本
- Twp sample t-test  双样本
	- Independent samples t-test  独立样本
	- Paired sample t-test 配对样本

#### One sample t-test

>_A one-sample Student's t-test is a location test of whether the mean of a population has a value specified in a null hypothesis._

是一种位置检验，检验总体均值是否具有原假设中指定的值。   
$$t = \frac{\bar x - \mu_0}{s/\sqrt{n}}$$   
- $\bar x$是样本均值
- s是样本标准差
- 此时自由度为n-1

从总体中，sample出一些样本，计算样本的均值$\bar x$，样本总体不需要满足正态分布，但是此时认为以样本均值为随机变量时，它遵循正态分布，以样本均值代表总体均值，比较其与假设值的距离  

根据中心极限定理，t统计量将服从近似正态分布  

#### Two sample t-tests
##### Independent samples

>_The independent samples t-test is used when two separate sets of independent and identically distributed samples are obtained, and one variable from each of the two populations is compared_

两组样本，独立且分布相同，比对两个总体的某一变量时使用

$$t = \frac{(\bar x_1 - \bar x_2)}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$  

##### Paired samples

>_Paired samples t-tests typically consist of a sample of matched pairs of similar units, or one group of units that has been tested twice (a "repeated measures" t-test)._

具有相似单元的匹配对的样本，或一组经过两次检验的单元    
两个样本之间是由依赖性的(dependent)。  


- 两组相似的样本
- 相依赖的随机变量是连续的
- 抽样是独立的
- 相依赖的变量近似正态分布

$$t = \frac{\bar d}{S_d/\sqrt{n}}$$   

- $\bar d$ 是配对的样本的差异的均值
- $S_d$是差异的标准差
- $n$是配对样本的数量



# Conclusion

实际上写下这些笔记的目的是基本明白t统计量和t检验，但是结果上并不是很清楚   
在不同t检验中，计算出来的t是否就是t统计量？只是因为情况不同计算不同而已？  
结论上看  
- 每种情况(单样本，线性回归最小二乘，双样本)都是计算估计值与目标值的距离，计算结果都是一种t分布
- 通过t分布来计算p值，进行t检验
- 在单样本中，通过样本标准差与根号n的比值估计均值的标准差，最小二乘则是标准误，非配对样本是标准差估计值估计的差异的标准差，配对样本则是差异的标准差估计值与根号n的比值(与单样本类似)

**数学知识还是差的太多**   

## Reference

- [Z-statistics vs. T-statistics | Inferential statistics | Probability and Statistics | Khan Academy - YouTube](https://www.youtube.com/watch?v=5ABpqVSx33I)
- [t-statistic - Wikipedia](https://en.wikipedia.org/wiki/T-statistic)
- [Student's t-test - Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test)
- [T Test (Students T Test) - Understanding the math and how it works - Machine Learning Plus](https://www.machinelearningplus.com/statistics/t-test-students-understanding-the-math-and-how-it-works/)
- [T-test - GeeksforGeeks](https://www.geeksforgeeks.org/t-test/)




  
