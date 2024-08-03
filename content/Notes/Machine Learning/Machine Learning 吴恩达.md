课程链接：[Machine Learning Specialization \[3 courses\] (Stanford) | Coursera](https://www.coursera.org/specializations/machine-learning-introduction)
# Supervised Machine Learning Regression and Classification
## 机器学习概述：
**机器学习(*Machine Learning*)：** 使计算机拥有学习能力而未经显式编程的学习。  

>  *from Arthur Samuel* 
>  "Field of study that gives computers the ability to learn without being expliciylty programmed"   


### 监督学习(Supervised learning)
- used most in real-world
- rapid advancements
#### Basis
##### Definition:  
**Supervised learning refers to algorithm that learn x $\rightarrow$ y (or input $\rightarrow$ output) mapping**  
Learns from given _right answer_ (the correct label _y_)  
从现成的xy映射中学习，并学会从x得到y。
##### Types
- **Regression(回归): predict a number from infinitely many possible outputs**
- **Classification(分类): predict categories(classes), a small finite limited set of possible output categories.** (强调可以非数字, 多种输入)
- 简而言之，监督学习学习输入与输出的正确映射关系，其中回归由输入在无限可能输出的数字中找输出，分类可多种输入，在有限可能输出(类别)中寻找输出。
### 非监督学习(Unsupervised learning)
#### Basis
##### Definition
**Find something interesting(paterns,structures...) in unlabeled data.**  
自己发现数据的规律  
##### Types
- **Clustering algorithm(聚类): cluster classification automatically**  
- **Anomaly detection(异常检测): detect unusual events**
- **Demensionality raduction(数据降维): compress a given big data set to a much smaller one while losing as little nformation as possible.**  
### Jupyter Notebooks
f format:
```python
# f format
print(f"adhdahhai{variable:.0f}adaij")
```
matplotlib magic:
```python
%matplotlib widget
```
### Terminology
- Training set: data used to train the model
### Notations
- $x$ : _input variable_, also called _feature or input feature_
- $y$ : _output variable_, also called _target variable_ or _output target_
- $m$ : total number of tranining examples
- $(x, y)$ : a a trainning example
- $(x^{(i)},y^{(i)})$ : $i^{th}$ traning example
  when multiple features, $x^i$ will be a list(vector), a row vector
- $\widehat{y}$ : the prediction of y
- $n$ : the number of features
## Regression
### Linear Regression
**Training set $\rightarrow$ learning algorithm $\rightarrow$ function\ or\ hypothesis\ or\ model**  
#### Basic model
$$f_{w,b}(x) = wx + b$$
**Linear ragression with one variable.** or **univariate linear regression.**  
- $w, b$ : the parameters of model, also referred to as coefficients(系数) or as weights(权重).  
#### Cost function(成本函数)
**first step to implement linear regression.**  
- Cost Function compares $\widehat{y}$ to the target $y$ :
	- by taking $\widehat{y} - y$  
		- the difference called error, representing how far the prediction is from the target.
	- then suqare it $(\widehat{y} - y)^2$ 
	- and sum up for calculate the the errors through the training set. $$\sum_{i=1}^m(\widehat{y}^{(i)} - y^{(i)})^2$$
	- avoid become larger with the bigger size training set, use average.$$\frac{1}{m}\sum_{i=1}^m(\widehat{y}^{(i)} - y^{(i)})^2$$
	- to neater$$\frac{1}{2m}\sum_{i=1}^m(\widehat{y}^{(i)} - y^{(i)})^2$$  

So that:
**The Cost Function(the squared error cost function):**
$$J(w,b) = \frac{1}{2m}\sum_{i=1}^m(\widehat{y}^{(i)} - y^{(i)})^2
$$
OR
$$J(w,b) = \frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)}) - y^{(i)})^2
$$
**Our goal is to chose a best cost function $J$ (or w and b):**  
对于Linear Regression，目标是  
$$minimise_{w,b}J(w,b)$$
	简化条件下，只考虑w的话，对每个w计算成本可得以下图像。  
	![[Pasted image 20240227163120.png]]
	当两个变量时，图像形如  
	![[Pasted image 20240227164656.png]]
	2D的三维图
	右边为contour plot(等高线图)
	![[Pasted image 20240227170139.png]]
	


#### 梯度下降法(Gradient Descent)
**To minimize any function**
##### Outline
- **start with guessed w, b (common set both to 0)
- **keep changing $w,b$ to reduce $J$
- **near ot at minimum**  
**principle:  use steepest descent(最速下降)，贪心，可能到达局部最小(local minimum).**  
>[!note]
>使用最小二乘法(the suqared error cost function)，结果一定是碗状
##### Formula 
$$w = w - \alpha \frac{\partial}{\partial w}J(w,b)$$
where: 
- $\alpha$ : the learning rate. 
	a small positive number between 0 and 1
	**control the step**(步长)
- 
- $\alpha \frac{\delta}{\delta w}J$ : the direction
and
$$b = b - \alpha \frac{\partial}{\partial b}J(w,b)$$
update parameters at the same time
$$\begin{cases} 
w = w - \alpha \frac{\partial}{\partial w}J(w,b) \\ 
b = b - \alpha \frac{\partial}{\partial b}J(w,b)
\end{cases}$$
当从左往右下降时，偏导的结果是负的，当从右往左下降时，偏导的结果是正的，所以是**减号**
![[Pasted image 20240302165906.png|291]]

where partitial derivative is
$$\begin{cases}
\frac{1}{m}\sum_{i=1}^m(f_{w,b}(x^{i})-y^{i})x^{i}&& (\partial w)\\
\frac{1}{m}\sum_{i=1}^m(f_{x,b}(x^i)-y^{i}) &&(\partial b) 
\end{cases} $$
>[!warning]
>同时迭代意味着上下的$w$都是$pre\_w$，都是没更新前的$w$和$b$
##### choose learning rate $\alpha$
- if too small: 
	will work but very slow
- too large:
	overshoot and won't work
    may fail to converge or may diverge (不收敛或者发散)
- when fall at a local minimum
    gradient descent can reach local minimum with fixed learning rate.  
    when near the local minimum, the derivative becomes smaller, update steps then become smaller, so it can reach minimum without changing the learning rate.
**注意越陡，偏导本身会越大**

##### Training
**repeat w and b until convergence.**  
算法，也就是公式本身保证了不断计算的情况，在合适的learning rate上，w,b的数值会越来越拟合样本，所以实现时只是规定了合理的迭代次数。
**a convex function(凸函数):  a bowl shape function and cannot have any local minimum. only center.**
**Batch gredient descent: the gredient descent proccess uses *all the training example*.**  batch:大概是批量的意思

#### The Code of linear regresssion
convention:
- $\frac{\partial J(w,b)}{\partial b}$  --  `dj_db`
- w.r.t -- with respect to
##### cost function compute code
```python
def compute_cost(X, y, w, b, verbose=False):
   """
   Compute the gradient for linear regression
    Args:
     X(ndarray (m,n)) : Data, m example with n features
     y(ndarray (m,)) : target values
     w(ndarray (n,)) : model parameters
     b(scalar)       : model parameter
     verbose(boolean): if true, print out intermediate value f_wb 
    Returns:
     cost(scalar)
   """
   # 用训练集特征一维获得个数
    m = X.shape(0)   
   # 计算 f_wb 
    f_wb = X @ w + b    # @是矩阵乘法
    total_cost = (1/(2*m)) * np.sum(f_wb-y)**2  # 平方     
    if verbose: print("f_wb:")
    if verbose: print(f_wb)    # 大概是为了分行

	return total_cost
```

##### gredient descent compute code
```python
def compute_gradient(X, y, w, b):
   """
   Compute the gradient for linear regression
    Args:
      X(ndarray (m,n)):data, m examples with n features
      y(ndarray (m,))  :target values
      w(ndarray (n,))  :parameter
      b(scalar)        :parameter
    Returns:
      dj_dw(ndarray (n,)): the gradient of the cost w.r.t the parameters w.
      dj_db(scalar)      : the gradient of the cost w.r.t the parameter b.
      
   """
    m = X.shape(0)
    f_wb = X @ w + b
    dj_dw = (1/m) * X.T@(f_wb - y)
    dj_db = (1/m) * np.sum(f_wb - y)
    
    return dj_dw, dj_db
```

>[!notation]
>1. 无论多少个特征，与参数w相乘后一定是向量
>2. 学会通过向量叉乘求和
##### optimize code(trainning) code
```python
def gradient_descent(x, y, w_in, b_in, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with learning rate alpha
    Args:
      x(ndarray (m,)) : Data, m examples
      y(ndarray (m,)) : target, m examples
      w_in,b_in(scalar) : initial values of model parameters
      alpha(float) : learning rate
      num_iters(int): number of iterations to run gradient
      cost_function: function to call to produce cost
      gradient_function:function to call to produce gradient
    Returns:
      w,b(scalar): the updated parameter    
      J_history(list): History of cost values
      p_history(list): History of parameters [w,b]
	"""
	w = copy.deepcoopy(w_in) # avoid modifying global w_in
	J_history = []
	p_history = []
	b = b_in
	w = w_in

	for i in range(num_iter):
		dj_dw,dj_db = gradient_function(x, y, w, b)
		w = w - alpha*dj_dw
		b = b - alpha*dj_db

		if i < 100000: # prevent resource exhaustion
			J_history.append(cost_function(x, y, w, b))
			p_history.append([w,b])
		if i % math.ceil(num_iter/10) == 0:
			print(f"Iteration{i:4}: cost {J_history[-1]:0.2e}",
				  f"dj_dw:{dj_dw: 0.3e},dj_db:{dj_db:0.3e}",
				  f"w{w:0.3e}, b:{b:0.5e}")
	return w, b, J_hitory, p_history
```

#### With Multiple Features/Variables
the more general model  
$$f_{w,b}(\textbf{x}) = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$
a simplied version
$$f_{\textbf{w},b}(\textbf{x}) = \textbf{w}·\textbf{x} + b$$
	where w and x is vector, the dot is matrix multiplication(or dot prodcut). 
	nemed multiple linear regression.  
#### Vectorization
就是利用语言特性，使用向量计算。  
python 本身有点乘运算  
numpy提供`.dot()`进行点乘，更快 
对于乘法  `f_wb = np.dot(w,b) + b`  
对于加减运算 `w = w - 0.1*d`  



#### gradient descents in multiple linear regression

$$\textbf{w} = \textbf{w}-\alpha \frac{\partial}{\partial \textbf{w}}J(\textbf{w},b)$$
$$b=b-\alpha \frac{\partial}{\partial b}J(\textbf{w},b)$$
$$\frac{\partial}{\partial \mathbf{w_j}}J(\mathbf{w},b) = \frac{1}{m}\sum_{i=1}^{m}(f_{\mathbf{w},b}(\mathbf{x_i}) - y)x_{ij}$$

#### Alternative to gradient descent
**Normal equation**
- **only for linear regression**
- solve for w,b without iteration
- 当特征数量太大时会很慢
- 可能有些库使用，但是自己做没用

#### The code for multiple linear regression routines
1. 构造模型
2. 建立成本函数，计算成本
3. 构建训练算法（梯度下降）
	1. 计算梯度
	2. 计算成本
4. 初始化数据
5. 训练

```python
import copy,math
import numpy as np
import matplotlib.pyplot as plt
# reduced display precision on numpy arrays
np.set_printoptions(precision=2)  
# 竖排是一种特征的样例，横排是不同特征
X_train = np.array([[2104, 5, 1,45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# 模型
def prediction(x,w,b):
	
	p = np.dot(x,w) + b
	return p 

# 计算cost function
def compute_cost(X, y, w, b):
	cost = 0.
	m = X.shape[0]
	for i in range(m):
		p_i = np.dot(X[i],w) + b
		cost_i = (p_i - y[i])**2
		cost = cost + cost_i
	cost = cost /(2*m)
	return cost

# 梯度下降

# 计算梯度
def compute_gradient(X, y, w, b):
	m,n = X.shape
	dj_dw = np.zeros(n,)
	dj_db = 0.
	
	for i in range(m):
		err = np.dot(X[i], w) + b - y[i]
		for j in range(n):
			dj_dw[j] = err*X[i,j]
		dj_db = dj_db + err
	dj_dw = dj_dw / m
	dj_db = dj_db / m
	
	return dj_dw, dj_db

# 训练模型函数
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
	J_history = []
	w = copy.deepcopy(w_in)
	b = b_in

	for i in range(num_iters):
		dj_dw, dj_db = gradient_function(X, y, w, b)
		b = b - alpha*dj_db
		w = w - alpha*dj_dw
		# 打印过程数据
		if i < 100000:
			J_history.append(cost_function(X, y, w, b))
		# 除以10后迭代到除以10的倍数，当num_iter大于10时一定输出十次，小于时迭代几次输几次
		if i % math.ceil(num_iter / 10) == 0:
			print(f"Iteration {i:4d}:Cost{J_history[-1]:8.2f}")

	return w, b, J_history

# 初始化
initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
# 训练
w_final, b_final, J_history = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

```




#### Feature Scaling(特征缩放/归一化)

当两个特征的数值量级差距较大时，大量级的特征容易主导模型，所以需要小的参数，小量级则相反  
说：而一开始的参数一般是没有明显的大小之分的，小量级比较可能需要大的参数，大量级比较可能需要小的参数，若都从0开始，大量级梯度比较大，小量级梯度比较小，跨度存在差距，大量级更容易到达终点，而小量级很难？

>[!think]
>所以问题是什么，优化不同步？还是这样比较难拟合（因为迭代次数比较难定？）
>
>答： 

[[Why Feature Scaling]]
##### methods
- **divide by the maximum**
	- range 0~1
- **mean normalization**
	- 均值归一化
	- usually range -1 ~ 1
	- calculate the mean value $\mu_i$
	- $$x_{ij}' = \frac{x_{ij} - \mu_i}{max-min}$$
	- 由于向量化和广播不需要每个都算
- **Z-score normalization**
	- Z-score 标准化
	- 可能得符合高斯分布？就是正态分布转标准化正态分布的式子
	- caculate standard deviation $\sigma$ and mean $\mu$ 
	- $$x_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
理想是归一化为 -1 ~ 1，但是不需要这么严格  
- 接近这个范围就可以
- 太小太大就不行




#### Tell if gradient descent is convergence(收敛)
![[Pasted image 20240316165002.png|300]]

**$J(w,b)$ should decrease after every iteration.** 
>[!notation]
>成本的可视化可以帮助确认学习率的选择是否正确或者有无bug

**Automatica convergence test**
- set $\epsilon = 10^{-3}$ (不一定是这个数)，**不如画图**
- if $J(w,b)$ decrease by $\le$ $\epsilon$ in one iteration, declare it convergence  
- which means that with w, b, cost function has reach the global mnimum value

 >[!Choosing learning rate]
 >小技巧？
 >画成本函数图发现不对，设置一个相当小的learning rate，如果还是不对，大概就是bug。如果对了，就稍微放大learning rate
 >**设置(0.001或者更小)，3倍放大，控制迭代数量，画图观察**
 
 
 
 
   
   


#### The code of Feature scaling and learning rate

##### 数据集离散图
```python
# 对各个特征分开画图，判断是否需要scaling
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey = ture)
for i in range(len(ax)):
	ax[i].scatter(X_train[:,i], y_train)
	ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()
```
##### 成本随迭代次数变化
`print(f"{i:0.2e})` 表示小数点后保留2个十进制位，e表示科学计数法。  
这个代码没有参考价值，而且还是我自己乱改的，真要写记得把历史存成字典（会比较简单？），然后直接画图。  
```python
def plot_cost_i(X, y, hist):
	fig, ax = plt.plot(figsize=(12, 3))
	ax.plt(hist["iter"], (hist["cost"]))
	ax.set_title("Cost vs Iteration")
	ax.set_xlabel("iteration")
	ax.set_ylabel("Cost")
	plt.show()
```

##### Z-score normalization
```python
def z_score_normalization(X):
	"""
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray): Shape (m,n) input data, m examples, n features
      
    Returns:
      X_norm (ndarray): Shape (m,n)  input normalized by column
      mu (ndarray):     Shape (n,)   mean of each feature
      sigma (ndarray):  Shape (n,)   standard deviation of each feature
    """
    # mean of each column
    mu = np.mean(X, axis=0)
    # standard deviation of each column
    sigma = np.std(X, axis=0)
    # element-wise calculation
    X_norm = (X - mu) / sigma

	return (X_norm, mu, sigma)
```

正则化后，等高线图会变得对称  
正则化还会加快训练  


# Advanced Learning Algorithm
# Unsupervised Learning recommenders reinforcement learning