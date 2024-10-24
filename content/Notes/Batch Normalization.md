---
tags:
  - DL
  - ML
---

training deep neural networks with tens of layers is challenging as they can be sensitive to the **initial random weights** and **configuration of the learning algorithm.**    

one possible reason is the distribution of the inputs to the deep layers may change after each mini-batch when the weights are updated(深层的输入分布随着权重改变发生变化). This can cause the learning algorithm to forever chase a moving target. (不断改变的输入分布难以找到和目标分布的正确映射)   
the change in the distribution of inputs to layers in the networks is referred to the name "**internal covariate shift(内部协变量偏移)**"(在时间上的偏移而不是在维度上的偏移)  

> Generally speaking, covariate shift is the case that the changed inputs(the distribution shift around) lead to the need of retraining the network.

Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch.(在一个batch内进行归一化)  

>batchnorm constrain the inputs to keep the same mean and standard deviation at least in a batch

It has the effect of **stabilizing the learning process** and **dramatically reducing the number of training epochs required**.  

### Problem of Training Deep Networks

>_Very deep models involve the composition of several functions or layers. The gradient tells how to update each parameter, under the assumption that the other layers do not change. In practice, we update all of the layers simultaneously._    ---- [Deep Learning](https://amzn.to/2NJW3gE)

粗略地理解，梯度作为参数更新方向是没有考虑其他层的更新的。梯度更新当前层假设前一层的输出不会再发生改变，输入分布不变去拟合输出分布。  

**the update procedure is forever chasing a moving target**  

>_This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities._   ---- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

about [[Saturating Non-linearities]]

存在的问题：
1. 参数的更新没有考虑其他层的更新
2. 网络内部的输入随权重更新改变分布
实际上是同个问题，参数更新没有协调，各自进行各自的调整，各处分布也各自改变，导致无法真正实现对目标函数的拟合。  
### Standardize Layer Inputs

Batch normalization is proposed as a technique to help coordinate the update of mulitiple layers in the model.  

It does this scaling the output of the layer, specifically by **standardizing** the activations of each input variable per mini-batch, such as the activations of a node from the previous layer.(对每一层的激活函数输出进行标准化)  

This process is also called “**whitening**” when applied to images in computer vision.  

Batchnorm has the effect of stabilizing and speeding-up the training process of deep neural networks. Especially for CNN and network with sigmoidal nonliearites   

Although **reducing “internal covariate shift”** was a motivation in the development of the method, there is some suggestion that instead batch normalization is effective because it **smooths and, in turn, simplifies the optimization function** that is being solved when training the network.   

### How to standardize  
original standardization is implemented during training by calculating the mean and standard deviation of each input variable to a layer **per mini-batch**  

$$
\mu = \frac{1}{m}\sum_iz^{(i)}
$$  
$$
\sigma^2 = \frac{1}{m}\sum_i(z^{(i)}-\mu)^2
$$  
$$
z_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$  
$$
\widetilde{z}^{(i)} = \gamma z_{norm}^{(i)}+\beta
$$  
- Why $\varepsilon$? 
	- It is added for numerical stability and is an arbitrarily small constant. 
- Why $\gamma$ and $\beta$ ?
	- restore the representation power of the network
	- In practice it is common to allow the layer to **learn two new parameters**, namely a new mean and standard deviation, Beta and Gamma respectively, it allow automatic scaling and shifting of the standardized layer inputs. These parameters are learned by the model as part of the training process.  


Given the choice of activation function, the distribution of the inputs to the layer may be quite non-Gaussian. It may be better to standardize the summed activation before activation function.（一些激活函数的输出可能会非高斯，此时选择在激活函数之前标准化可能更好）  
#### Improvement
but if batch size is too small, or mini-batches do not contain a representative distribution of the training set, or differences in the standardized inputs between training and inference can result in differences in performance.   
This can be solved by **Renormalization**.  

>Batch Renormalization extends batchnorm with a per-dimension correction to ensure that the activations match between the training and inference networks.  ---- [Batch Renormalization](https://arxiv.org/abs/1702.03275) 

### Tips for using batch normalization

1. **Use with different network types**
	1. such as MLP, CNN, RNN
2. **Probably use before the Activation**
	1. may be used before or after the activation function in the previous layer
	2. **after** if for s-shaped   -- 和李宏毅老师课上说的不同
	3. **before** if for activation function that may result in non-Gaussian  --- modern default
	4. **but it depends**
3. **Use Large Learning Rates**
	1. batchnorm smooth the training, so it need much larger learning rates
4. **Less Sensitive to Weight Initializaiton**
5. **Alternate to Data Preparation**
	1. Batchnorm could be used to standardize **raw input variables** that have differing scales.
	2. the batch size must be sufficiently representative of the range of each variable
	3. if variables have highly non-gaussian distribution, it will be better to preform data scaling as a pre-processing step.  如果特征的分布相当非高斯的话，最好在数据预处理的时候进行规范化而不是在训练时。
6. **If Use With Dropout**  👈 **It depends**
	1. batchnorm offers some regularization effect, reducing generalization error, perhaps no longer requiring dropout for regularization
		1. Each mini-batch is scaled by the mean/variance computed on just that mini-batch
		2. it **adds some noise**(计算出来的均值和方差并不代表整个数据集) to the values whithin that batch
		3. causes **a slight regularization**
		4. the larger the batch is, the less noise it will has, and the regularization will be reducing as well
	2. random dropout may cause noisy to normalization
7. **At test Time, use the exponatially weighted average across mini-batch as the final mean and standard deviation**  



### References
- [A Gentle Introduction to Batch Normalization for Deep Neural Networks](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)
- [Why Does Batch Norm Work? deeplearning.ai- YouTube](https://www.youtube.com/watch?v=nUUqwaxLnWs)
- [Batch normalization - Wikipedia](https://en.wikipedia.org/wiki/Batch_normalization)
- [Batch Normalization - OpenAI- YouTube](https://www.youtube.com/watch?v=Xogn6veSyxA)  👈没看
- [\[D\] Batch Normalization before or after ReLU? : r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/) 👈 see? it is controversial
