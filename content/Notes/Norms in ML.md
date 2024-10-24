---
tags:
  - DL
  - ML
---

## Overview
> almost copy from [deep learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?ie=UTF8&qid=1472485235&sr=8-1&keywords=deep+learning+book)

In machine learning, norm is a function that measure the size of vector  

The $L^p$ norm is given by  
$$
||\textbf{x}||_p = (\sum_i|x_i|^p)^{\frac{1}{p}}
$$  

for $p \in \mathbb{R}$, $p \ge 1$    

Intuitively, the norm of a vector $\textbf{x}$ measures the distance from the origin to the point $\textbf{x}$     

Rigorously, a norm is any function $f$ that satisfies the following properties:
- $f(x) = 0 \Rightarrow \textbf{x} = 0$
- $f(x = y) \le f(x) + f(y)$ (**triangle inequality**)  
- $\forall \alpha \in \mathbb{R}, f(\alpha\textbf{x}) = |\alpha|f(\textbf{x})$

The $L^2$ norm, with $p = 2$, is known as the **Euclidean norm(欧几里得范数)**. It is simply the Euclidean distance.   
$$||\textbf{x}|| = \sum_ix_i^2$$
$L^2$ norm is used frequentily in ML, and often denoted simply as $||\textbf{x}||$  
It is more common to used its squared version, which can be calculated simply by $\textbf{x}^T\textbf{x}$  

squared $L^2$ norm is more convenient to work with mathematiocally and computaionally, for example, each derivative of the squared $L^2$ norm only depend on the corresponding element of $\textbf{x}$, while $L^2$ norm depond on the entire vector  

In other contexts, the squared $L^2$ norm may be undesirable because it increases slowly near the origin.   
It maybe important to discriminate between elements that are exactly zero and elements that are small but nonzero, and thus we turn to $L^1$ norm  
$L^1$ norm increase at the same rate(linear) in all location, but retains mathematical simplicity  
$$
||\textbf{x}||_1 = \sum_i|x_i|
$$  
$L^1$  norm can be used as a substiture for the number of nonzero entries   

other norm may needed include **max norm** $L^{\infty}$ norm  
$$
||\textbf{x}||_{\infty} = max_i|x_i|
$$  
It is the maximum absolute value of the components of the vector.  

In the context of deep learning, the most common way to measure the size of a matrix is **Frobenius norm**:  
$$
||A||_F = \sqrt{\sum_{i,j}A^2_{i,j}}
$$  
a analogous to $L^2$ norm  

by the way, we can use norm to rewrite the dot product  
$$
\textbf{x}^T\textbf{y} = ||\textbf{x}||_2||\textbf{y}||_2cos\theta
$$  

## Regularization with Norm

### From PRML

> 3.1.4  Regularized least squares

Error function with regularization term  
$$
E_D(\textbf{w})+\lambda E_W(\textbf{w})
$$  

where $\lambda$ is the regularization coefficient that controls the relative importance of the data-dependent error $E_D(\textbf{w})$ and the regulatization term $E_W(\textbf{w})$.  

Regularization aim to cotrol over-fitting, so that Considering adding regularizaiton term on the parameters is naturally    

We can use norm to describe the amount or other property of parameters.  

Squared $L^2$ norm   
$$
E(\textbf{w}) = \frac{1}{2}\textbf{w}^T\textbf{w}
$$  
the $1/2$ here is added for later convenience  
the entire penalty become  
$$
\frac{\lambda}{2}\textbf{w}^T\textbf{w}
$$  
This particular choice of regularizer is known in the context of machine literature as **weight decay(权重衰减)**, it encourages weight values to decay towards zero(due to the minimize optimal), unless supported by data.  
It provides an example of  a **parameter shrinkage(参数缩减)** (in the context of statistic), lead to a closed form of the objective function   

A more general regularizer is sometimes used  
$$
E(\textbf{w})+ \frac{\lambda}{2}\sum_{j=1}^M|w_j|^q
$$  

since the norm describe the distance distance from the origin to the point $\textbf{x}$, It can be visulized like  
![Pasted image 20241024144101](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020241024144101.png)
 

In other perspective, minimizing objetive function with norm($L^1$ norm or $L^2$ norm) can be calculated using Lagrange multipilers.   
in that way, minimize objective function is equivalent ti minimizing the cost function subject to constraint   
$$
\sum_{j=1}^M|w_j|^q \le \eta
$$  
It can also be visualized like the picture above  

$L^1$ norm would lead to a sparse model in which the corresponding basis functions play no role(系数置0，对应的基函数无效)   

![Pasted image 20241024145819](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020241024145819.png)    

>the solution is $\textbf{w}^*$.  
>$L^2$ norm in the left and $L^1$ norm in the right  
>It shows that it $L^1$ (lasso) lead to a sparse solution in which one parameter is 0  
>It should also be recognized that with regularization, the parameters both decreased  








