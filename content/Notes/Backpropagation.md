---
tags:
  - ML
  - DL
---

## 李宏毅

### Gradient Descent
参数的更新    
$$\theta^1 = \theta^0 - \eta\nabla L(\theta^0)$$
$$\theta^2 = \theta^1 - \eta\nabla L(\theta^1)$$
但是有上百万的参数，高效的计算 ---> **backpropagation**  

> 所以backpropagation就是梯度下降

#### 数学基础

**Chain Rule**  微分链式法则  
![[Pasted image 20240627103304.png|425]]
![[Pasted image 20240627103329.png|425]] 

#### Backpropagation

**Lost function**
$$L(\theta) = \sum_{n=1}^NC^n(\theta)$$
- $C^n$ 是对每一笔**x**(数据)定义的cost function，加和作为当前输出和目标的距离  

求梯度  
$$\frac{\partial L(\theta)}{\partial w} = \sum_{n=1}^N \frac{\partial C^n(\theta)}{\partial w}$$
以线性神经元为例  
对于单独一个线性神经元  有  
$$z = \sum_i^nw_ix_i + b $$  
z作为输入，传给激活函数
$$a = \sigma(z)$$  
经过多层后输出$y$，与 $\hat y$ 计算距离 $C$
$$C = \hat y - y$$
$$\frac{\partial C}{\partial w} = \frac{\partial z}{\partial w}\frac{\partial C}{\partial z}$$

- $\frac{\partial z}{\partial w}$ 称为 forward pass  
- $\frac{\partial C}{\partial z}$ 称为 backward pass  

##### Forward pass
$$\partial z/\partial w_i = x_i$$
结果就是各权重对应的输入  
该权重的偏微分数值就是前一个神经元的输入  
![[Pasted image 20240628085611.png|475]]

##### Backward pass
计算 $$\frac{\partial C}{\partial z}$$  
z算完出来  
过激活函数  
$$a = \sigma(z)$$  
$$\frac{\partial C}{\partial z} = \frac{\partial a}{\partial z}\frac{\partial C}{\partial a}$$  
$$\partial a/ \partial z = \sigma'(z)$$  
如果是sigmoid  
![[Pasted image 20240628090219.png|350]]   

一层一层加  
再加一层  
![[Pasted image 20240628090317.png|500]]  
这个时候，C看成是$z'$ 和$z''$ 的某种加和  (**chain rule**)
$$\frac{\partial C}{\partial a} = \frac{\partial z'}{\partial a}\frac{\partial C}{\partial z'} + \frac{\partial z''}{\partial a}\frac{\partial C}{\partial z''}$$
> 这里只考虑了两个神经元，如果是多个就是多个链式相加
> 而且到这里都是在计算第一层权重系数中的某一个权重的梯度


$$\frac{\partial z'}{\partial a} = w_3 \ \ \ \ \frac{\partial z''}{\partial a} = w_4$$  
到这里
$$\frac{\partial C}{\partial z} = \sigma'(z)[w_3\frac{\partial C}{\partial z'} + w_4\frac{\partial C}{\partial z''}]$$  
假设后面两个偏导已经算出来了   
此时可以有另外的角度看待这个式子    
![[Pasted image 20240628093745.png|450]]
$\frac{\partial C}{\partial z'}和\frac{\partial C}{\partial z''}$  看成是输入$x_1, x_2$ ，经过加和后乘上激活函数$\sigma'(z)$  得到输出就是 $\frac{\partial C}{\partial z}$  
以微分的方式逆过来看  
- **但是 $\sigma'(z)$ 已经在前面forward pass 决定了，所以现在是一个常数**  
然后  
$$\frac{\partial C}{\partial z'} \  形式和\  \frac{\partial C}{\partial z}\ 一致$$  
所以$\frac{\partial C}{\partial z'}$也是一个逆向的神经元输出，通过下一层的输入z的偏微分可以求出前一层的偏微分，最终给出权重的偏微分(**compute recursively**)    
一直到输出层(output layer)  
$$\frac{\partial C}{\partial z_{out}} = \frac{\partial y}{\partial z_{out}}\frac{\partial C}{\partial y}$$  
$y$ 是激活函数输出，所以$\frac{\partial y}{\partial z_{out}}$ 就是激活函数的导在$z_{out}$ 上的取值  
$\frac{\partial C}{\partial y}$ 看损失函数怎么定义，直接对y求偏导就可以得到  
所以秒算(李宏毅)  
所以从最后一层开始往前算，一直到第一层就能得到梯度  
所以此时变成  
![[Pasted image 20240628101000.png|550]]  

计算前一层的偏微分，就是后一层加权和，然后通过激活函数的导进行放大  
- 输入是前向的
- 是个常数

##### 总结
backpropagation 是梯度下降的算法，通过两个部分结合，逆向的计算梯度  
backpropagation （$\partial C / \partial w$）分为两个部分  
一个是forward pass，一个是backward pass
![[Pasted image 20240628103353.png|600]]
- forward pass ($\partial z / \partial w$)   就是指前一个激活函数输出到下一个激活函数输入的过程，也就是z，如何对参数求导的问题
	- **结论是 每一层输入对参数的导都是前一层的输出**
- backward pass ($\partial C / \partial z$) 是计算损失对输入的导的过程，将计算过程逆过来考虑，通过最后一层反向计算总体的损失对参数的导
	- 这样的计算产生了激活函数导数的相乘(vanishing gradient problem)
	- 反向和正向是等效的，只是这样比较有效且不复杂
- 两部分合起来构成了反向传播(相乘)





## 自己的补充
- [ ] 待补




## reference
- [ML Lecture 7: Backpropagation - YouTube](https://www.youtube.com/watch?v=ibJpTrp5mcE)
- [Understanding Backpropagation Algorithm | by Simeon Kostadinov | Towards Data Science](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
- [A Comprehensive Guide to the Backpropagation Algorithm in Neural Networks](https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide)