---
tags:
  - ML
---

## PCA

>_A notes from Deep Learning Foundations and Concepts (Christopher M. Bishop, Hugh Bishop)_

Principal component analysis, or PCA, is widely used for applications such as dimensionality reduction, lossy data compression, feature extraction, and data visualization. It is also known as the Kosambi–Karhunen–Loeve transform.    

**PCA** can be defined as **the linear proejction that maximizes the variance of the projecrted data**, or it can be defined as **the linear projection that minimizes the average projection cost(hte mean squared distance between the data points and their projections)**   

Given a dataset, PCA seeks a space of lower dimensionality(known as **principal subspace**) that satisfies the definitions above.  

![Principal20Analysis20principal](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Principal20Analysis20principal.gif)  

### Maximum variance formulation

Consider a data set of observations ${x_n}$ where $n = 1,...,N$, and $x_n$ is a variable with dimensionality D.  
Our goal is to project the data onto a space having demensionality $M < D$, while maximizing the variance after projected  
> assume that $M$ is given  

When $M = 1$, we can define the direction of the space using a $D$-dimensional vector $\textbf{u}_1$, and without loss of generality, we choose $\textbf{u}_1$ that satisfies $\textbf{u}_1^T \textbf{u}_1 = 1$  
then projection of $\textbf{x}_n$ will be $\textbf{u}_1^T \textbf{x}_n$  
the mean of the procjected data is $\textbf{u}_1^T \bar {\textbf{x}}$   

$$
\overline{{{{{\bf x}}}}}=\frac{1} {N} \sum_{n=1}^{N} {{\bf x}}_{n} 
$$

the variance will be 
$$
\frac{1} {N} \sum_{n=1}^{N} \left\{\mathbf{u}_{1}^{\mathrm{T}} \mathbf{x}_{n}-\mathbf{u}_{1}^{\mathrm{T}} \overline{{{{\mathbf{x}}}}} \right\}^{2}=\mathbf{u}_{1}^{\mathrm{T}} \mathbf{S} \mathbf{u}_{1} 
$$
where $S$ is the data covariance matrix  
$$
\mathbf{S}=\frac{1} {N} \sum_{n=1}^{N} ( \mathbf{x}_{n}-\overline{{{{\mathbf{x}}}}} ) ( \mathbf{x}_{n}-\overline{{{{\mathbf{x}}}}} )^{\mathrm{T}}. 
$$  
And then maximize the projected variance $\mathbf{u}_{1}^{\mathrm{T}} \mathbf{S} \mathbf{u}_{1}$   
To prevent $||u_1|| \rightarrow \infty$, A appropriate constraint comes from the normalization condition $\bf{u}_1^\mathrm{T} \bf{u}_1 = 1$   
Introduce a Lagrange multiplier denoted by $\lambda_1$   
then the formula becomes  
$$
{\bf u}_{1}^{\mathrm{T}} {\bf S} {\bf u}_{1}+\lambda_{1} \left( 1-{\bf u}_{1}^{\mathrm{T}} {\bf u}_{1} \right). 
$$  
Setting the derivative with respect to $\bf{u}_1$ equal to zero, there is a stationary point when    
$$
{\bf S u}_{1}=\lambda_{1} {\bf u}_{1}, 
$$  

It says that $\bf{u}_1$ must be an eigenvector of $\bf {S}$. If left-multiply by $\bf{u}_1^\mathrm{T}$, and with the condition $\bf{u}_1^\mathrm{T} \bf{u}_1 = 1$    
the variance will be   
$$
\mathbf{u}_{1}^{\mathrm{T}} \mathbf{S} \mathbf{u}_{1}=\lambda_{1} 
$$

**Maximizing the variance becomes choosing the eigonvector $\bf{u}_1$ that has the largest eigonvalue $\lambda_1$. This eigonvector is kown as the first principal component**  

We can then define additional principal components in an incremental fashion(增量) by choosing each new direction to be that which maximizes the projected variance amongst all possible directions orthogonal to those already considered. (取所有与当前选择的方向正交的最大化方差的方向)   

In the case of $\mathrm{M}$-dimensional projection space, the optimal solution would be the $\mathrm{M}$ eigonvectors of the covariance matrix $\mathrm{S}$ that have the $\mathrm{M}$ largest eigenvalues   

总结来说，找到$\mathrm{M}$维的投影空间，先计算均值，以均值计算协方差矩阵，求解协方差矩阵的$\mathrm{M}$个最大特征值对应的特征向量。   
### Minimum-error formulation

Introduce a **complete orthonormal** set of D-dimensional basis vectors $\{\bf{u}_i\}$ where $i \le D$ that satisfy  
$$
\mathbf{u}_{i}^{\mathrm{T}} \mathbf{u}_{j}=\delta_{i j}. 
$$

each dataset can be represented exactly by a linear combination of the basis vectors  

$$
\mathbf{x}_{n}=\sum_{i=1}^{D} \alpha_{n i} \mathbf{u}_{i} 
$$

It can be regarded as a rotation of the coordinate system to a new system defined by the $\{\bf{u}_i\}$  

Taking the inner product with $\bf{u}_j$, we can obtain     
$$
\mathbf{x}_{n}^\mathrm{T} \bf{u_j}=\left(\sum_{i=1}^{D} \alpha_{n i} \mathbf{u}_{i}\right)^{\mathrm{T}} \bf{u}_j  
$$

and since orthonormality   
we can obtain $a_{nj} = \mathbf{x}_{n}^\mathrm{T} \bf{u_j}$, and sothat we can write  

$$
\mathbf{x}_{n}=\sum_{i=1}^{D} \left( \mathbf{x}_{n}^{\mathrm{T}} \mathbf{u}_{i} \right) \mathbf{u}_{i}. 
$$

However, we use D-dimentional space for the expression, our goal is to represent the data in $M < D$ subspace.  
We can approximate the data by  

$$
\widetilde{\mathbf{x}}_{n}=\sum_{i=1}^{M} z_{n i} \mathbf{u}_{i}+\sum_{i=M+1}^{D} b_{i} \mathbf{u}_{i} 
$$
where the $\{z_{ni}\}$ depend on the particular data point, and $\{b_i\}$ are constants  
We are free to choose $\{\bf{u}_i\}$, the $\{z_{ni}\}$ and the $\{b_i\}$ so as to minimize the projection error(Introduce by reduction in dimensionality)  

The error can be defined as  

$$
J=\frac{1} {N} \sum_{n=1}^{N} \| \mathbf{x}_{n}-\widetilde{\mathbf{x}}_{n} \|^{2}. 
$$  
Substituting into all formula, setting the derivative with respect to $z_{nj}$ to zero, and making use of the orthonormality conditions, we obtain   

$$
z_{nj} = \bf{x}_n^{\mathrm{T}}\bf{u}_j
$$  

And respect to $b_i$ to zero, we obtain  

$$
b_{j}=\overline{{{{\mathbf{x}}}}}^{\mathrm{T}} \mathbf{u}_{j} 
$$

where $j > M$  

Substitute for $z_{nj}$ and $b_i$, the difference between data and projection becomes  

$$
\mathbf{x}_{n}-\widetilde{\mathbf{x}}_{n}=\sum_{i=M+1}^{D} \left\{( \mathbf{x}_{n}-\overline{{{\mathbf{x}}}} )^{\mathrm{T}} \mathbf{u}_{i} \right\} \mathbf{u}_{i} 
$$
We can see the minimum error is given by the orthogonal projection  
We therefore obtain the error purely of the $\{\bf{u}_i\}$ in the form    


$$
J=\frac{1} {N} \sum_{n=1}^{N} \sum_{i=M+1}^{D} \left( \mathbf{x}_{n}^{\mathrm{T}} \mathbf{u}_{i}-\mathbf{\overline{{{x}}}}^{\mathrm{T}} \mathbf{u}_{i} \right)^{2}=\sum_{i=M+1}^{D} \mathbf{u}_{i}^{\mathrm{T}} \mathbf{S} \mathbf{u}_{i}. 
$$  
Aim to avoid $u_i = 0$, we must add a constraint to the minimization  

For a intuition about the result, let's consider a case that $D = 2$, and $M = 1$   
By adding Lagrange multiplier $\lambda_2$, we minimize  

$$
\widetilde{J}={\bf u}_{2}^{\mathrm{T}} {\bf S} {\bf u}_{2}+\lambda_{2} \left( 1-{\bf u}_{2}^{\mathrm{T}} {\bf u}_{2} \right) 
$$

It is the same as the maximum variance process, we just obtain the minimum instead  
By setting the derivative with respect to $\bf{u}_2$ to zero, we obtain $S\bf{u}_2 = \lambda_2\bf{u}_2$  
back-subtitude it into $J$, we obtain  $J = \lambda_2$  
With the goal to minimize $J$, we choose the smaller eigonvalue, and **therefore we choose the eigonvector corresponding to the larger eigonvalue as the principal subspace**   

The general solution is  
$$
S\bf{u}_i = \lambda_i\bf{u}_i
$$  
And $J$ is given by  
$$
J = \sum_{i = M+1}^D\lambda_i
$$

We choose the $D − M$ smallest eigenvalues, adn **hence the eigenvectors defining the principal subspace are those corresponding to the $M$ largest eigenvalues.**  

## PLSR

> notes for [16 Partial Least Squares Regression | All Models Are Wrong: Concepts of Statistical Learning](https://allmodelsarewrong.github.io/pls.html)

Initially, I wanted to learn about Partial Least Squares, but I found that it might be too broad or challenging.  So just start from PLSR first   

PLS method has a big family, and PLSR may be a friendly one  
PLSR is another dimension reduction method to regularize a model, like PCR, PLSR seek subspace describe by $\bf{z_1}, ... \bf{z}_k$(or linear combinations of $\bf{X}$ )  

Moreover, there is an implicit assumption: $\bf{X}$ and $\bf{y}$ are assumed to be functions of a reduced $(k < p)$ number of components $\bf{Z}$ that can be used to decompose them  

$$
\mathbf{X}=\mathbf{Z} \mathbf{V}^{\mathsf{T}}+\mathbf{E} 
$$
![Pasted image 20241103185941](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020241103185941.png)  

$$
\bf{y} = \bf{Z}b + \bf{e}
$$

![Pasted image 20241103190019](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020241103190019.png)  

So the model in the $X$-space is   
$$
\bf \hat {X} = \bf{Z}\bf{V}^ \mathrm{T}
$$

and y becomes  
$$
\bf \hat y = \bf{Z}\bf{b}
$$

#### But How to get the components

Assume that both the inputs and the response are mean-centered(and possibly standardized). Compute the covariances between all the inputs variables and the reponse:    

$$
\mathbf{\tilde{w}}_{1}=( c o v ( \mathbf{x}_{1}, \mathbf{y} ), \ldots, c o v ( \mathbf{x}_{p}, \mathbf{y} ) ). 
$$  

In vector-matrix notation   
$$
\bf \tilde{w}_1 = \bf X^{\mathrm{T}}\bf y
$$

> In fact, the notations above are not the same.   
> The matrix notation larger than the covariance  
> But we care about the direction only, and their direction will be the same if inputs and response are mean-centered(their mean is 0)  
> so it doesn't matter    

and we can also rescaled $\bf \tilde w_1$ by regressing each predictor $\bf x_j$ onto $\bf y$   

$$
\tilde{\mathbf{w}}_{\mathbf{1}}=\mathbf{X}^{\mathsf{T}} \mathbf{y} / \mathbf{y}^{\mathsf{T}} \mathbf{y} 
$$

and then normalize $\bf \tilde w_1$  
$$
\bf w_1 = \frac{\bf \tilde w_1}{||\bf \tilde w_1||}
$$

We use these weights to compute the first component $\bf z_1$   

$$
\mathbf{z}_1 = w_{11}\mathbf{x}_1 + ...+w_{p1}\mathbf{x}_p
$$  
since $\bf w_1$ is unit-vector, the fomula can be expressed like  

$$
\mathbf{z}_{1}=\mathbf{X} \mathbf{w}_{1}=\mathbf{X} \mathbf{w}_{1} / \mathbf{w}_{1}^{\mathsf{T}} \mathbf{w}_{1} 
$$

Then we use the component to regress inputs onto it to obtain the first **PLS loading**  

$$
{\bf v_{1}}={\bf X}^{\mathsf{T}} {\bf z_{1}} / {\bf z_{1}^{\mathsf{T}} {\bf z_{1}}} 
$$

regress response onto it to obtain the first PLS coeffs  

$$
b_{1}=\mathbf{y}^{\mathsf{T}} \mathbf{z}_{1} / \mathbf{z}_{1}^{\mathsf{T}} \mathbf{z}_{1} 
$$

And thus we have a first one-rank approximation  
$$
\hat{\mathbf{X}} = \bf z_1 v_1^{T}
$$
Then we can obtain the rasidual matrix $\bf X_1$ (named **deflation**)   

$$
\mathbf{X}_{1}=\mathbf{X}-{\hat{\mathbf{X}}}=\mathbf{X}-\mathbf{z}_{1} \mathbf{v}_{1}^{\mathsf{T}} \qquad{\mathrm{( d e f l a t i o n )}} 
$$
deflate the response  
$$
\mathbf{y}_1 = \mathbf{y} - b_1\mathbf{z}_1
$$

This is the first round, and we can obtain k components by repeating the process k times, iteratively  

every time we reduce the rasidual by obtain the approximation of the previous rasidual matrix and the target $\bf y$  
the effect of these approximation are synthesized by matrix multiply  

> 个人认为，这么做的目的是通过回归的方式来拟合，得到降维子空间的同时减小产生的误差(每次都是拟合误差)，这些拟合的结果最终出现在矩阵中(开头的式子)，效果被综合

What PLS is doing is calculating all the different ingredients (e.g. $\bf w_i, z_i, v_i$, $b_i$ ) separately, **using least squares regressions. Hence the reason for its name partial least squares.**  


## References

### PCA
- [Deep Learning - Foundations and Concepts](https://www.bishopbook.com/)
- [Principal Component Analysis (PCA) Explained | Built In](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
### PLSR
- [16 Partial Least Squares Regression | All Models Are Wrong: Concepts of Statistical Learning](https://allmodelsarewrong.github.io/pls.html)
- [Partial least squares regression - Wikipedia](https://en.wikipedia.org/wiki/Partial_least_squares_regression)
### Difference

**Not yet**
- [Principal Component Analysis (PCA) and Partial Least Squares (PLS) Technical Notes](https://docs.tibco.com/pub/stat/14.0.0/doc/html/UsersGuide/GUID-F02AD2C4-E4C1-4E55-95A9-4E18F3D4B72D.html)


