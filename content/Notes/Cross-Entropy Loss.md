cross-entropy is a measure from the field of information theory, generally calculating the difference between two probability distributions  

[[KL divergence(KL 散度)]] calculate the relative entropy between two probability distributions, while cross-entropy calculate the total entropy  

## Cross-Entropy

Information quantifies the number of bits required to encode and transmit an event. (信息量化了编码和传递事件需要的位数)  

**Lower probability events(surprising) have more information, higher probability events(unsurprising) have less information.**    

information $h(x)$ for an event $x$ can be calculated as follows   
$$h(x)=-log(P(x))$$  
***Entropy* is the number of bits required to transmit a randomly selected event from a probability distribution.**  熵用于描述分布，是传输一个随机从概率分布中选取的事件需要的位数  

A **skewed distribution(偏态分布) has a low enrtopy(less surprise)**, whereas a **distribution where events have equal probabilty(事件等概率) has a larger entropy**   
熵衡量了一个分布的随机性(信息量的大小)   


- Low Probability Event(surprising): more information.  
- Higher Probability Event (unsurprising): Less information.
- Skewed Probability Distribution (unsurprising): Low entropy.
- Balanced Probability Distribution (surprising): High entropy.

Entropy $H(x)$ can be calculated by(if discrete states)  
$$H(x) = -\sum_xP(x)log(P(x))$$  
**熵是信息的期望**  

***Cross-entropy* calculates the number of bits required to represent or transmit an event from one distribution compared to another distribution.(when use another distribution)**  
当使用一个分布描述另一个分布的事件时需要的熵(信息/位数的期望)  

>_… the cross entropy is the average number of bits needed to encode data coming from a source with distribution p when we use model q …_     ---- Machine Learning: A Probabilistic Perspective

Cross-entropy between two probability distributions, such as Q from P, can be stated as   
$$H(P,Q)$$  

Cross-entropy $H(P,Q)$ can be calculated as  
$$H(P,Q) = -\sum_xP(x)*log(Q(x))$$  
the result will be **positive**, meaning **the bits required when use $Q$ to represent event in $P$**   

if the distribution is the same,(two probability distributions are identical), the result is the entropy   

>log is the **base-2 lograithm**(对数), meaning that the results are in **bits**.  
>if the **base-e(natural logarithm)** is used instead, the result will have the units called **nats**


## Cross-Entropy Versus KL Divergence

Cross-Entropy measures the total number of bits needed to represent message with $Q$ instead of $P$, wheras KL-divergence measures the extra bits required.   
交叉熵是使用另一个分布描述时需要的所有位数，而KL散度是额外的位数  

KL Divergence can be calculated by  
$$KL(P||Q) = -\sum_x P(x)log\frac{Q(x)}{P(x)}$$  
> _In other words, the KL divergence is the average number of extra bits needed to encode the data, due to the fact that we used distribution q to encode the data instead of the true distribution p._   
>  ----- Machine Learning: A Probabilistic Perspective

KL-divergence is often referred to as the **"relative entropy"**  

- **Cross-Entropy: Average number of total bits to represent an event from Q instead of P.**
- **Relative Entropy (KL Divergence): Average number of extra bits to represent an event from Q instead of P.**

Cross-entropy can be calculated by KL-divergence  
$$H(P, Q) = H(P) + KL(P||Q)$$  
the same to KL-divergence  

$$KL(P||Q) = H(P,Q) - H(P)$$  
KL-dibergence and Cross-entropy is not symmetrical  
$$H(P,Q) \ !=H(Q,P)$$  

## How to Calculate Cross-Entropy

> most of framework have built-in cross-entropy function  

Cross-entropy can be implemented by  
```python
def cross_entropy(p, q):
	return -sum([p[i]*log2(q[i]) for i in range(len(p))])
```

calculated with KL-divergence  
```python
def kl_divergence(p, q):
	return -sum([p[i] * log2(q[i]/p[i]) for i in range(len(p))])

def entropy(p):
	return -sum([p[i]*log2(p[i]) for i in range(len(p))])

def cross_entropy(p, q):
	return entropy(p) + kl_divergence(p, q)
```

## Cross-Entropy as Loss Function

Cross-entropy is widely used as a loss function when optimizing classification models.  

>_… using the cross-entropy error function instead of the sum-of-squares for a classification problem leads to faster training as well as improved generalization._  
>   ----- Pattern Recognition and Machine Learning, 2006

使用Cross-entropy的分类问题将数据是否属于某个类别转化为概率分布的拟合问题，寻找一个符合类别定义的概率分布  

类别定义  
- Random Variable: the example for which we require a predicted class label.(样例数据为随机变量)
- Events: Each class label that could be predicted.(是否属于某个类别为事件)
- Each example has a probability of 1.0 for the known class label, and 0.0 for other labels.(每个样例对应的类别概率为1，其他类别概率为0)  

the model estimate the probability of an example belonging to each class label and then Cross-entropy can then be used to calculate the difference between the two probability distributions. (模型预测出样例类别的概率分布，然后交叉熵计算与目标概率分布的差别)  


the target probability distribution for an input as the class label 0 or 1 interpreted as "impossible" and "certain" respectively. That means no surprise and they have no infotmation/zero entropy  
对于目标分布来说，这些样例(数据)是确定的，信息为0(熵为0)  

> for all data used when training, they have zero entropy, which means the value of Cross-entropy become the difference between two distribution(用交叉熵逼近目标分布就变成最小化交叉熵)

With C represents the class set, the cross-entropy can be calculated by   

$$H(P,Q) = -\sum_i^{|C|}P(C_i)log(Q(C_i))$$


the base-e or natrual logarithm is used for classification tasks, this means units are in nats  

The use of cross-entropy for classification often gives different specific names based on the number of classes  
- **Binary Cross-Entropy**: Cross-entropy as a loss function for a binary classification task.
- **Categorical Cross-Entropy**: Cross-entropy as a loss function for a multi-class classification task.

mirroring the name of the classification task
- **Binary Classification**: Task of predicting one of two class labels for a given example.
- **Multi-Class/Categorical Classification**: Task of predicting one of more than two class labels for a given example.


Using **one hot encoding**, for an example that has a label for the first class, the probability distribution can be represented as $[1,0,0]$ if there are three classes.  

>one hot encoding 在这里是一种类别的编码方式

This probability distribution has no information as the outcome is certain. Therefore the entropy for this variable is zero.  

Therefore, a cross-entropy of 0.0 when training a model indicates that the predicted class probabilities are identical to the probabilities in the training dataset, e.g. zero loss.  

in this context, using KL-divergence is the same as cross-entropy  

>In practice, **a cross-entropy loss of 0.0 often indicates that the model has overfit the training dataset**, but that is another story.

## Intuition for Cross-Entropy

When the predicted distribution become far away from target gradually, the cross-entropy between predicted distribution and target distribution increases as well.  

![[Pasted image 20240810214419.png]]  

#### What is a good cross-entropy score?
It depends  
In general  

- **Cross-Entropy = 0.00** Perfect probabilities.(overfit may occur)
- **Cross-Entropy < 0.02** Great probabilities.
- **Cross-Entropy < 0.05** On the right track.
- **Cross-Entropy < 0.20** Fine.
- **Cross-Entropy > 0.30** Not great.
- **Cross-Entropy > 1.00** Terrible.
- **Cross-Entropy > 2.00** Something is broken.

## Cross-Entropy versus Log Loss
they are not the same, but calculate the same quantity when used as loss functions for classification problems  
不是同个东西但是实际上是等价的  

#### Log loss
Logistic loss refers to the loss function commonly used to optimize a logistic regression model.  

Log loss come from **Maximum Likelihood Estimation(MLE)**  

That involves selecting a likelihood function that defines how likely a set of observations are given model parameters(给定模型参数时测值发生的可能性)  

Because it is more common to **minimize a function than to maximize it in practice**, the log likelihood function is inverted by adding a **negative sign to the front**. This transforms it into a **Negative Log Likelihood** function or **NLL** for short.  

log loss = negative log-likelihood, under a Bernoulli probability distribution (二分类时的negative log-likelihood)  


## References
- [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) 👈 **mainly**
- [Cross Entropy Loss | Machine Learning Theory](https://machinelearningtheory.org/docs/Unified-View/cross-entropy/)
- [Deep Learning More Techniques](https://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Deep%20More%20(v2).ecm.mp4/index.html)

