---
tags:
  - Math
  - statistic
---

## Overview

Degree of Freedom is a fundamental concept that represents the number of independent values or pieces of information that can vary in a statistical analysis without violating any constraints.   

It is typically calculated by  
$$DF = n - p$$  
where  
- n is the sample size
- p is the parameters being estimated(number of restrictions)

for example, in t-tests. the degree of freedom is one less than the sample size, as one parameter(the mean) is estimated from the data  

## Deeper?

> The number of independent pieces of information used to calculate th statistic is called the degrees of freedom[^1]  

When represent how values are related to each other, it would introduce some retrictionsk and so that not all the pieces of information can vary freely  

Degrees of freedom change the shape of the null distribution.  

#### In t-distribution

when $df = 1$,  the distribution is leptokurtic(fat-tailed)  
as df increases, the distribution become narrower(more and more similiar to normal distribution)  
when $df \ge 30$, t distribution is almost the same as the standard normal distribution   

#### In Chi-square distribution
[[Chi-square test]] 👈 TODO
when $df < 3$, the probability distribution is shaped like a backwards "J"  
when $df \ge 3$, it will become hump-shaped(驼峰状), with the peak of the hump located at $df - 2$, and the hump is right-skewed  
when $df > 90$, chi-square is approximated by a normal distribution  


## Reference

[^1]: [How to Find Degrees of Freedom | Definition & Formula](https://www.scribbr.com/statistics/degrees-of-freedom/)

> Some text are written by AI
