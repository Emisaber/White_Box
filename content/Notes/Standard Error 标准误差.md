---
tags:
  - Math
  - statistic
---

# Standard Error

> _The standard error(SE) of a statistic is the standard deviation of its sampling distribution or an estimate of that standard deviation_

标准误是样本分布的标准差，或者它的估计值  
如果统计量是样本均值，则称为均值的标准误(**SEM**)   

数学上，样本均值的方差会等于**样本总体的方差除以样本量**，随着样本量的增加，样本均值会更紧密地聚集在总体均值周围，即随着样本量的增加，样本均值方差会趋近于0   

**则均值的标准误等于标准差除以样本量的平方根，衡量了样本均值围绕总体均值的离散程度**   

> 应该可以认为，标准误可以衡量抽样统计量对假设值(真实值)的离散程度

### Standard error of the sample mean
SEM  


$$
\sigma_{\bar X} = \frac{\sigma}{\sqrt{n}}
$$  
$\sigma$是总体的标准差，n是样本数量   

>要缩小k倍误差，n(试验次数)需要增大到$k^2$倍

通常总体的$\sigma$不可知，使用样本的方差代替  

$$
\sigma_{\bar X} \approx \frac{\sigma_X}{\sqrt{n}}
$$  
有可能样本的方差也不知道，可以使用样本方差的估计值  

$$
\sigma_{\bar X} \approx \frac{S_X}{\sqrt{n}}
$$  

