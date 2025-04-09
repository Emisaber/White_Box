---
tags:
  - ML
  - DL
  - LLM
---
## What are Diffusion Models

>_GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAE relies on a surrogate loss. Flow models have to use specialized architectures to construct reversible transform._  
>from [What are Diffusion Models? \| Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
>感觉是精辟的总结  


>[!notes]
>#### Some background needed  
>##### Markov chain
>马尔科夫链(markov chains)是一个数学模型，根据某些概率规则从一个状态转移到另一个状态，它的特点是马尔可夫性质(markov property)，转化(transition)到未来任意状态的概率只依赖于当前的状态，与过去的状态无关，称 无记忆性  (memorylessness)  即，马尔可夫系统不保存过去信息   
>
>##### non-equilibrium thermodynamics
>非平衡态热力学是热力学的一个分支，研究系统在远离平衡态时的行为  
>扩散模型中正向扩散近似于非平衡态热力学中的熵增过程，反向扩散近似于熵减的过程

扩散模型定义了一个不断增加随机噪声的马尔可夫链，模型学习如何reverse the diffusion process来从噪声中构造想要的数据   
扩散模型的训练过程是固定的，在高维的空间(与原数据一致)中训练(而没降维)   

![Pasted image 20250316151605](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250316151605.png)  

#### Forward diffusion process 前向过程

逐步增加高斯噪声的马尔可夫链   

给定样本x，逐步增加高斯噪声，总共执行T步，产生一系列的noisy sample $x_1, x_2, ...x_T$。Step size受varianve schedule $\beta_t \in (0, 1)$  
$\beta_t$ 一般随着步数增加，直到最后一步，样本趋近于纯噪声(标准正态分布)  

数学表达式为   

$$
q ( \mathbf{x}_{t} | \mathbf{x}_{t-1} )={\mathcal{N}} ( \mathbf{x}_{t} ; {\sqrt{1-\beta_{t}}} \mathbf{x}_{t-1}, \beta_{t} \mathbf{I} ) \quad q ( \mathbf{x}_{1 : T} | \mathbf{x}_{0} )=\prod_{t=1}^{T} q ( \mathbf{x}_{t} | \mathbf{x}_{t-1} ) 
$$
- 第一个式子表明单步转移概率分布(转移到$x_t$状态的概率)是服从 均值为$\sqrt{1-\beta_t}$ 和 方差为$\beta_tI$的正态分布
- 第二个式子即时马尔可夫链的基本形式

这样的定义我们可以sample任意时刻的 $x$  (**reparameterization trick**)   
![Pasted image 20250316160132](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250316160132.png)   
其中  $\alpha_{t}=1-\beta_{t} \; \mathrm{a n d} \; \bar{\alpha}_{t}=\prod_{i=1}^{t} \alpha_{i}$   
$\beta_i$ 逐渐增大，$\bar \alpha$ 逐渐减小  

>为什么能这么算呢  
>实际上动笔硬算就行   
>两个高斯分布相加，得到  $\mathcal{N} ( \mathbf{0}, ( \sigma_{1}^{2}+\sigma_{2}^{2} ) \mathbf{I} ).$  
>计算标准差为  $\sqrt{( 1-\alpha_{t} )+\alpha_{t} ( 1-\alpha_{t-1} )}=\sqrt{1-\alpha_{t} \alpha_{t-1}}.$
>注意此时标准差第二个有展开后得到的加权

##### Connection with stochastic gradient Langevin dynamics(朗之万动力学)

stochastic gradient Langevin dynamics can produce samples from a probability density $p(x)$ using only the gradients $\nabla_{\mathbf{x}} \operatorname{l o g} p ( \mathbf{x} )$  in a Markov chain of updates   

$$
{\bf x}_{t}={\bf x}_{t-1}+\frac{\delta} {2} \nabla_{\bf x} \operatorname{l o g} p ( {\bf x}_{t-1} )+\sqrt{\delta} \epsilon_{t}, \quad\mathrm{w h e r e} \ \epsilon_{t} \sim{\cal N} ( {\bf0}, {\bf I} ) 
$$
其中 $\delta$ 是 step size，当 $T \rightarrow \infty, \epsilon \rightarrow 0$的时候，$x_T$ 等于真实的概率密度 $p(x)$   

这种梯度更新法加入了Gaussian noise避免了陷入local minima中   

>[!note]  
>###### 为什么出现了这个？   
>朗之万动力学是一种从复杂分布中采样的方法，通过确定性(梯度)和随机性(噪声)，x 逐渐趋近于目标分布 $p(x)$  
>Diffusion的反向过程可以看成是一种相近的形式(随机微分方程SDE角度)  
>这一角度，Diffusion 可以被描述为一种 score-based 模型



#### Reverse diffusion process

如果我们能reverse forward的过程(从$q(x_{t-1}|x_t)$中抽样)，我们就能从高斯噪声中重构图像。如果$\beta_t$ 足够小，$q(x_{t-1}|x_t)$仍然是高斯分布。  
如果 $q(x_{t-1}|x_t)$ 仍然是高斯分布，它应满足  

$$
q( {\bf x}_{t-1} | {\bf x}_{t} )={\cal N} ( {\bf x}_{t-1} ; \boldsymbol{\mu} ( {\bf x}_{t}, t ), \boldsymbol{\Sigma} ( {\bf x}_{t}, t ) ) 
$$

假若数据皆可访问，我们能通过上面的trick直接得到分布  
但是现实数据是不能全部访问的，所以我们不能直接得到$q(x_{t-1}|x_t)$。因此我们需要有一个模型 $p_{\theta}$ 来预测这个均值与方差  
数学表达式如下  

$$
p_{\theta} ( {\bf x}_{0 : T} )=p ( {\bf x}_{T} ) \prod_{t=1}^{T} p_{\theta} ( {\bf x}_{t-1} | {\bf x}_{t} ) \quad p_{\theta} ( {\bf x}_{t-1} | {\bf x}_{t} )={\cal N} ( {\bf x}_{t-1} ; \mu_{\theta} ( {\bf x}_{t}, t ), \mathbf{\Sigma}_{\theta} ( {\bf x}_{t}, t ) ) 
$$


如果给定 $x_0$ (conditioned $x_0$)，$q(x_{t-1}|x_t, x_0)$ 是比较容易处理的   

$$
q ( {\bf x}_{t-1} | {\bf x}_{t}, {\bf x}_{0} )={\cal N} ( {\bf x}_{t-1} ; \tilde{\mu} ( {\bf x}_{t}, {\bf x}_{0} ), \tilde{\beta}_{t} {\bf I} ) 
$$
有如下数学推导  

根据Bayes' rule 有   
![Pasted image 20250316185807](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250316185807.png)   


> [!notes]
> ##### about Bayes' rule
> 贝叶斯公式    
> $$ 
> P ( A | B )={\frac{P ( B | A ) \, P ( A )} {P ( B )}} 
> $$
>$P(A)$ 称为A的先验概率，$P(B)$为B的先验概率，$P(A|B)$是后验概率，$P(B|A)$称为A和B的似然性  
>则贝叶斯公式描述为  
>$$
>后验概率 = (先验概率 \times 似然性)/标准化常数
>$$
>贝叶斯公式可扩展到两个以上的变量  
>$$
>P ( A | B, C )=\frac{P ( A ) \, P ( B | A ) \, P ( C | A, B )} {P ( B ) \, P ( C | B )} 
>$$
>

>补一点说明  
>根据两个变量的贝叶斯公式，A为 $x_{t-1}$，B为 $x_0$，C为 $x_t$ ，将 $P(A)P(B|A)/P(B)$ 转为条后验概率 $P(A|B)$ 则可以得到一开始的式子  
>再往后的推导基于高斯分布，具体的不是很懂，目标是求出当前高斯分布 $q(x_{t-1}|x_t, x_0)$ 的均值与方差  

根据高斯分布的形式，忽视 $C(x_t, x_0)$(与 $x_{t-1}$ 无关，影响不大) 化简二次项得到   

$$
\tilde{\beta}_{t}=1 / ( \frac{\alpha_{t}} {\beta_{t}}+\frac{1} {1-\bar{\alpha}_{t-1}} )=1 / ( \frac{\alpha_{t}-\bar{\alpha}_{t}+\beta_{t}} {\beta_{t} ( 1-\bar{\alpha}_{t-1} )} )=\frac{1-\bar{\alpha}_{t-1}} {1-\bar{\alpha}_{t}} \cdot\beta_{t} 
$$


$$
\begin{aligned} {{\tilde{\mu}_{t} ( \mathbf{x}_{t}, \mathbf{x}_{0} )}} & {{} {{}=( \frac{\sqrt{\alpha_{t}}} {\beta_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}}} {1-\bar{\alpha}_{t-1}} \mathbf{x}_{0} ) / ( \frac{\alpha_{t}} {\beta_{t}}+\frac{1} {1-\bar{\alpha}_{t-1}} )}} \\ {{}} & {{} {{}=( \frac{\sqrt{\alpha_{t}}} {\beta_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}}} {1-\bar{\alpha}_{t-1}} \mathbf{x}_{0} ) \frac{1-\bar{\alpha}_{t-1}} {1-\bar{\alpha}_{t}} \cdot\beta_{t}}} \\ {{}} & {{} {{}=\frac{\sqrt{\alpha_{t}} ( 1-\bar{\alpha}_{t-1} )} {1-\bar{\alpha}_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}} \bar{\beta}_{t}} {1-\bar{\alpha}_{t}} \mathbf{x}_{0}}} \\ \end{aligned} 
$$
实际上，DDPM的做法并不在意方差 $\widetilde \beta$，DDPM 固定方差（相比于学习方差来尝试优化效果） 
所以重点在  ${{\tilde{\mu}_{t} ( \mathbf{x}_{t}, \mathbf{x}_{0} )}}$    
根据前面的trick，有 $\mathbf{x}_{0}=\frac{1} {\sqrt{\bar{\alpha}_{t}}} ( \mathbf{x}_{t}-\sqrt{1-\bar{\alpha}_{t}} \mathbf{\epsilon}_{t} )$  
则  
$$
\begin{array} {l} {{{\tilde{\mu}_{t}=\frac{\sqrt{\alpha_{t}} ( 1-\bar{\alpha}_{t-1} )} {1-\bar{\alpha}_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}} {1-\bar{\alpha}_{t}} \frac{1} {\sqrt{\bar{\alpha}_{t}}} ( \mathbf{x}_{t}-\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t} )}}} \\ {{{=\frac{1} {\sqrt{\alpha_{t}}} \left(x_t - \frac{1-a_t} {\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{t} \right)}}} \\ \end{array} 
$$

我们希望模型预测出分布的均值，我们需要让模型预测结果接近 ${{\tilde{\mu}_{t} ( \mathbf{x}_{t}, \mathbf{x}_{0} )}}$    

>到这里我们可以跳步，因为 $\widetilde \mu_t$ 的各个变量，只有 $\epsilon$ 是非确定的，其余都是模型预测前给定的，所以我们只要让模型预测这个 $\epsilon$ ，即噪声就行。至于如何计算损失，随便引入一个MSE直觉上是合理的，DDPM的结果也是如此。   
>只是我们可能需要一个更make sense 的推导，来选择一个符合目标( $\mathrm{argmax}\ logp_\theta(x)$ )的损失

回到目标上，根据ELBO技巧，有   
$$
\begin{align*}
L_{CE} &= -\mathbb{E}_{q(x_0)} \log p_{\theta}(x_0) \\
&= -\mathbb{E}_{q(x_0)} \log \left( \int p_{\theta}(x_{0:T}) dx_{1:T} \right) \\
&= -\mathbb{E}_{q(x_0)} \log \left( \int q(x_{1:T} | x_0) \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} | x_0)} dx_{1:T} \right) \\
&= -\mathbb{E}_{q(x_0)} \log \left( \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} | x_0)} \right] \right) \\
&\le -\mathbb{E}_{q(x_{0:T})} \log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} | x_0)} \\
&= \mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_{1:T} | x_0)}{p_{\theta}(x_{0:T})} \right] = L_{VLB}
\end{align*}
$$

我们得到VLB，然后将VLB的各项推导成tacklable的形式  
有   
![Pasted image 20250404094519](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250404094519.png)   

为了简单描述，将各项重写为  

$$
\begin{array} {c} {{{L_\mathrm{V L B}=L_{T}+L_{T-1}+\cdots+L_{0}}}} \\ {{{\mathrm{w h e r e} \; L_{T}=D_{\mathrm{K L}} ( q ( \mathbf{x}_{T} | \mathbf{x}_{0} ) \parallel p_{\theta} ( \mathbf{x}_{T} ) )}}} \\ {{{L_{t}=D_{\mathrm{K L}} ( q ( \mathbf{x}_{t} | \mathbf{x}_{t+1}, \mathbf{x}_{0} ) \parallel p_{\theta} ( \mathbf{x}_{t} | \mathbf{x}_{t+1} ) ) \mathrm{~ f o r ~} 1 \leq t \leq T-1}}} \\ {{{L_{0}=-\operatorname{l o g} p_{\theta} ( \mathbf{x}_{0} | \mathbf{x}_{1} )}}} \\ \end{array} 
$$
可以看到 $L_T$ 是固定值($q(x_T|x_0)$ 是正向加噪过程，固定；而 $p_\theta(x_T)$ 是反向的起点，即得到Gaussian noise $x_T$ 的分布，也就是一个标准正态分布)，$L_0$ 在原论文中使用 seperate discrete decoder 处理  

我们重点关注 $L_t$   
$L_t$ 所有项都是两个分布的KL 散度，其中 $q(x_t|x_{t+1}, x_0)$ 是我们上文推导得到的反向真实分布，而 $p_\theta(x_t|x_{t+1})$ 是模型预测的反向过程   
对于 $q(x_t|x_{t+1}, x_0)$ 有分布   
$$
q ( {\bf x}_{t-1} | {\bf x}_{t}, {\bf x}_{0} )={\cal N} ( {\bf x}_{t-1} ; \frac{1} {\sqrt{\alpha_{t}}} \left(x_t - \frac{1-a_t} {\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{t} \right), \tilde{\beta}_{t} {\bf I} ) 
$$  
对于 $p_\theta(x_t|x_{t+1})$ 有分布  
 $$
{\bf x}_{t-1}={\cal N} ( {\bf x}_{t-1} ; \frac{1} {\sqrt{\alpha_{t}}} \Big( {\bf x}_{t}-\frac{1-\alpha_{t}} {\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{\theta} ( {\bf x}_{t}, t ) \Big), \Sigma_{\theta} ( {\bf x}_{t}, t ) )
$$  
我们可以得到  
$$
\begin{align*}
L_t &= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2||\Sigma_\theta(x_t, t)||^2} ||\tilde{\boldsymbol{\mu}}_\theta(\mathbf{x}_t, t) - \boldsymbol{\mu}(\mathbf{x}_t, t)||^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2||\Sigma_\theta||^2} || \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right) - \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) ||^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) ||\Sigma_\theta||^2} || \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) ||^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) ||\Sigma_\theta||^2} || \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) ||^2 \right]
\end{align*}
$$

>复杂的推导不是很会，这里应是将两个分布的方差当成是一致的，则KL 散度等价于两个均值的加权平方差(直觉理解确实)

DDPM 在训练的时候简化了这个损失，改成  

$$
\begin{aligned} {{{L_{t}^{\mathrm{s i m p l e}}}}} & {{} {{} {{}=\mathbb{E}_{t \sim[ 1, T ], \mathbf{x}_{0}, \epsilon_{t}} \Big[ \| \boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta} ( \mathbf{x}_{t}, t ) \|^{2} \Big]}}} \\ {{{}}} & {{} {} {{} {{}=\mathbb{E}_{t \sim[ 1, T ], \mathbf{x}_{0}, \epsilon_{t}} \Big[ \| \boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta} ( \sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \boldsymbol{\epsilon}_{t}, t ) \|^{2} \Big]}}} \\ \end{aligned} 
$$

最终的损失化简为   
$$
L_{simple} = L_t^{simple} + C
$$


也就是训练伪代码中   
![Pasted image 20250404105249](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250404105249.png)   

生成部分的伪代码也可以理解为构造了 $x_{t-1}$ 的分布   
![Pasted image 20250404105348](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250404105348.png)  

#### Connection with noise-conditioned score networks(NCSN)

知识匮乏，看个大概   
有一种score-based generative model method，基于 Langevin dynamics，使用 数据分布的gradients从分布中采样。这个gradients没能真实得到，通过模型来进行估计。  
即  
$$
{\bf s}_{\theta} ( {\bf x} ) \approx\nabla_{\bf x} \operatorname{l o g} q ( {\bf x} ). 
$$
其中$s_\theta$ 是score network，梯度为 score 的定义   
根据 manifold hypothesis，尽管数据看上去是高维的，大多数数据都集中在低维的manifold中。为了在高维上scalable，有以下两种方法，一种是额外增加噪声(denoising score matching)，一种是随机投影(sliced score matching)  

其中denoising score matching的做法是 给 $q(\tilde x|x)$ 增加一个 预先定义的噪声，通过score matching 来拟合 $q(\tilde x)$  

在 Diffusion 的语境下   
$$
q ( { \bf x } _ { t } | { \bf x } _ { 0 } ) \sim { \cal N } ( \sqrt { \bar { \alpha } _ { t } } { \bf x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) { \bf I } )
$$  
$$
\mathbf{s}_{\theta} ( \mathbf{x}_{t}, t ) \approx\nabla_{\mathbf{x}_{t}} \operatorname{l o g} q ( \mathbf{x}_{t} )=\mathbb{E}_{q ( \mathbf{x}_{0} )} [ \nabla_{\mathbf{x}_{t}} \operatorname{l o g} q ( \mathbf{x}_{t} | \mathbf{x}_{0} ) ]=\mathbb{E}_{q ( \mathbf{x}_{0} )} \Big[-\frac{\epsilon_{\theta} ( \mathbf{x}_{t}, t )} {\sqrt{1-\bar{\alpha}_{t}}} \Big]=-\frac{\epsilon_{\theta} ( \mathbf{x}_{t}, t )} {\sqrt{1-\bar{\alpha}_{t}}} 
$$

所以如果通过 score based 角度来理解 diffusion 的话，  
- 在训练过程中，模型学习了 $\epsilon_\theta(x_t, t)$，实际上也就是隐式学习了各个阶段的score function
- 即diffusion通过训练一个 score network来拟合不同t的 $\nabla_{\mathbf{x}_{t}} \operatorname{l o g} q ( \mathbf{x}_{t} )$，然后依据  Langevin dynamics 逐步逼近真实分布 $q(x)$  
#### 优化

##### 对 $\beta_t$

DDPM中的 $\beta_t$ 是线性变化的，从 $10^{-4}$ 到 $0.02$   
[后续的论文](https://arxiv.org/abs/2102.09672)提出了对此的修改，将linear 改成  cosine-based variance schedule     
具体公式如下  

$$
\beta_{t}=\mathrm{c l i p} ( 1-\frac{{\bar{\alpha}}_{t}} {{\bar{\alpha}}_{t-1}}, 0. 9 9 9 ) \quad{\bar{\alpha}}_{t}=\frac{f ( t )} {f ( 0 )} \quad\mathrm{w h e r e ~} f ( t )=\operatorname{c o s} \Big( \frac{t / T+s} {1+s} \cdot\frac{\pi} {2} \Big)^{2} 
$$
s 是防止t = 0时过小的offset    

![Pasted image 20250405102639](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250405102639.png)    

具体的schedule函数的选择是任意的，只要满足两端缓慢变化和中间接近线性就可以   

##### 对 $\Sigma_\theta$  

DDPM中固定方差为 $\beta_t$ 和 $\tilde \beta_t$   
同改进论文中，提出学习方差   

方差定义为   

$$
\mathbf{\Sigma}_{\theta} ( \mathbf{x}_{t}, t )=\operatorname{e x p} ( \mathbf{v} \operatorname{l o g} \beta_{t}+( 1-\mathbf{v} ) \operatorname{l o g} \tilde{\beta}_{t} ) 
$$

现在的损失中是方差无关的，所以需要加入一个能够指导方差学习的项，论文使用  

$$
L_{\mathrm{h y b r i d}}=L_{\mathrm{s i m p l e}}+\lambda L_{\mathrm{V L B}} 
$$

其中 $L_{VLB}$ 只与方差有关，关于均值的部分不计算梯度。$\lambda = 0.001$     

然后说，论文发现这样很难算，最后使用一个time-averaging smoothed version  

具体见论文[\[2102.09672\] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)   


## Conditioned Generation

生成模型通常需要有conditioning information，充当图片的描述或标签  

### Classifier Guided Diffusion

为了将类别标签加入到diffusion process中   
[这篇论文](https://arxiv.org/abs/2105.05233) 训练了一个 classifier $f\phi(y|x_t,t)$   
生成过程加入了 $\nabla_{\mathbf{x}} \operatorname{l o g} f_{\phi} ( y | \mathbf{x}_{t} )$ 作为指导   
有score function   

$$
\begin{align*}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y) &= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y|\mathbf{x}_t) \\
&\approx -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_{\phi}(y|\mathbf{x}_t) \\
&= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \left( \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_{\phi}(y|\mathbf{x}_t) \right)
\end{align*}
$$

此时 predictor 需要预测的形式就变成   
$$
\bar{\boldsymbol{\epsilon}}_{\theta}(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_{\phi}(y|\mathbf{x}_t)
$$

为了控制classifier的影响强弱，可以加入一个weight $w$  
$$
\bar{\boldsymbol{\epsilon}}_{\theta}(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t}w \nabla_{\mathbf{x}_t} \log f_{\phi}(y|\mathbf{x}_t)
$$

得到 ablated diffusion model(ADM) 和有指导的 ADM-G   

DDPM sample 过程  
![Pasted image 20250405114316](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250405114316.png)   

DDIM 过程  
![Pasted image 20250405114327](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250405114327.png)


### Classifier Free Guidance(CFDM)

实际上，通过数学推导可以发现，并不需要一个显式的classifier来得到conditional diffusion    
classifier的分布梯度可以重新参数化为   

$$
\begin{aligned} {\nabla_{\mathbf{x}_{t}} \operatorname{l o g} p ( y | \mathbf{x}_{t} )} & {{} {} {{}=\nabla_{\mathbf{x}_{t}} \operatorname{l o g} p ( \mathbf{x}_{t} | y )-\nabla_{\mathbf{x}_{t}} \operatorname{l o g} p ( \mathbf{x}_{t} )}} \\ {{}} & {{} {{}=-\frac{1} {\sqrt{1-\bar{\alpha}_{t}}} \Big( \epsilon_{\theta} ( \mathbf{x}_{t}, t, y )-\epsilon_{\theta} ( \mathbf{x}_{t}, t ) \Big)}} \\ \end{aligned} 
$$  
记此时模型需要学习的均值为 $\bar \epsilon(x_t, t, y)$   

$$
\begin{aligned} {{\bar{\epsilon}_{\theta} ( \mathbf{x}_{t}, t, y )}} & {{} {{}=\epsilon_{\theta} ( \mathbf{x}_{t}, t, y )-\sqrt{1-\bar{\alpha}_{t}} \ w \nabla_{\mathbf{x}_{t}} \operatorname{l o g} p ( y | \mathbf{x}_{t} )}} \\ {{}} & {{} {{}=\epsilon_{\theta} ( \mathbf{x}_{t}, t, y )+w \big( \epsilon_{\theta} ( \mathbf{x}_{t}, t, y )-\epsilon_{\theta} ( \mathbf{x}_{t}, t ) \big)}} \\ {{}} & {{} {{}=( w+1 ) \epsilon_{\theta} ( \mathbf{x}_{t}, t, y )-w \epsilon_{\theta} ( \mathbf{x}_{t}, t )}} \\ \end{aligned} 
$$
通过调节参数 $w$ ，可以控制有条件和无条件生成图片影响的比例   

真实的训练过程通过随机地消除条件 $y$ 来使模型能够生成无条件图像   

>[!note]
>###### 为什么要训练无条件下的图像(直观解释)
>公式 $\bar{\epsilon}_{\theta} ( \mathbf{z}_{\lambda}, \mathbf{c} )=( 1+w ) \epsilon_{\theta} ( \mathbf{z}_{\lambda}, \mathbf{c} )-w \epsilon_{\theta} ( \mathbf{z}_{\lambda} )$ 可以重写为 $\epsilon_{\mathrm{g u i d e d}}=\epsilon_{\mathrm{c o n d}}+( s-1 ) ( \epsilon_{\mathrm{c o n d}}-\epsilon_{\mathrm{u n c o n d}} )$  (换元整理)   
>如果从这个角度来看，混合两种图像相当于在原本无条件图像上加上有条件图像的影响。这样的话，无条件下的图像相当于基线，模型生成时在此基础上加上条件的影响。这样的做法可能缘于模型对条件并不敏感，s 可以调控这一点   

在[GLIDE](https://arxiv.org/abs/2102.09672) 论文中，比较了两种策略(CLIP guidance 和 Classifier-free)，发现CFDM的表现更好一些。论文认为在CLIP策略中，模型可能hack CLIP，欺骗CLIP来获得高分，导致表现偏弱   

## Speed up Diffusion

原始Diffusion的生成相当缓慢，reverse process的 T 可能是几千步。  

>For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.

一种可行的方法是进行跳步。限制采样步数 S ，每 $[T/S]$ 步进行采样。此时采样的步数为 ${t_1, t_2, ..t_s}$， $S< T$。  
例如$T=1000$，我们指定 $S = 100$，每十步进行一次更新，得到更新步数为 $10,20,..1000$。模型直接预测对应步数的噪声进行更新。  
这样的做法通过牺牲质量换取了时间。   

### DDIM

**fewer sampling steps**   

我们可以重写 $q(x_{t-1}|x_t, x_0)$    
$$
\begin{aligned}
\mathbf{x}_{t-1} &= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t-1}-\sigma_t^2}  \boldsymbol{\epsilon}_t + \sigma_t \boldsymbol{\epsilon} \\
&= \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_{\theta}^{(t)}(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2 } \boldsymbol{\epsilon}_{\theta}^{(t)}(\mathbf{x}_t) + \sigma_t \boldsymbol{\epsilon}
\end{aligned}
$$
>引入超参数 $\sigma$ ，第二项前后的分布没有发生变化，则前后 $x_{t-1}$ 是等价的  
>把整个式子看成是对 $\epsilon$ 的变化，则可以得到 $x_{t-1}$ 的分布  

$$
q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_{\theta}^{(t)}(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \sigma_t^2 \boldsymbol{\epsilon}_{\theta}^{(t)}(\mathbf{x}_t), \sigma_t^2 \mathbf{I} \right)
$$
从式子上看，$\sigma$ 是无所谓大小的，在DDPM的分布中  

$$
\tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
$$
我们引入一个 $\eta$ ，设 $\sigma = \eta \tilde \beta$，则当 $\eta = 1$ 时，为DDPM，是一个马尔可夫链过程，此时每一个 $t-1$ 状态依赖于上一个时间步的采样。  
而如果 $\eta = 0$，分布的随机性消失，我们得到一个非马尔可夫链过程。这个过程是确定性的(没有方差)，即给定一个初始的高斯噪声，依据公式最终得到的图像是一致的，这样的方法为DDIM   
由于这样的确定性/非马尔可夫，我们没有必要一步步采样，可以通过跳步来加速图像生成。  

实际效果参考如下(FID score)  
![Pasted image 20250407102047](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407102047.png)  

在实际上生成中，每一次生成经过两个步骤  
- 依据当前的 $x_t$ 估计 $x_0$ 
- 根据公式更新 $x_t$  


>[!note]
>##### 为什么DDIM需要多次采样
>既然反向去噪过程是一个确定性的函数，为什么需要多次采样  
>直觉上理解大概是  
>- 尽管反向是确定的，但是 $x_0$ 不是已知的，通过估计来进行采样，如果一步跨度过大，误差也会增大
>- 尽管给定 时间步表和初始噪声，模型生成结果会是确定的，但是并不见得这个结果是最优的，通过调整时间步可能得到更好的结果

相比于 DDPM，DDIM 有以下优势   
- 能用更少的步骤生成高质量图片
- 拥有一致性，由于DDIM的确定性，样本 conditioned on the same latent variable 会有相近的高层特征(高层语义上相近)
	- 相同的噪声输入会有相近的结果
- 一致性使得，DDIM在latent variable中的插值会对应语义上平滑过渡的生成结果
	- 如进行线性插值 $x_T''=\alpha x_T = (1-\alpha)x_T'$ ，生成图像会在语义上连续变化

DDIM实际上不完全是一种模型，DDIM重新参数化了反向过程(转化为非马尔可夫过程)，可以被理解为广义的扩散模型框架。DDIM的采样方法适用了DDPM方法训练的模型，用于加速采样。    

### Progressive Distillation

通过蒸馏训练好的 deterministic sampler(DDIM)，训练 student DDIM，使得 studet DDIM的每一步都等效于 teacher DDIM 的两步  
![Pasted image 20250407110106](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407110106.png)   

具体算法为  
![Pasted image 20250407110125](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407110125.png)  

### Consistency Models

又是一个新模型，再说   

### Latent Variable Space

[Latent diffusion model(LDM)](https://arxiv.org/abs/2112.10752) 在 latent space 中进行图像处理(加噪去噪)   
LDM 发现图像的大多数bits都起感知细节上的作用，图像的语义和概念性的部分在经过大程度的压缩后仍然能够保留。所以LDM不在pixel space 上进行图像去噪而是在latent space中，降低了运算量     

LDM的做法有两个阶段  
- perceptual compression process
- diffusion process

在perceptual compression process阶段，LDM使用一个autoencoder $\mathcal{E}$ 压缩输入图像 $x \in \mathbb{R}^{H \times W \times 3}$ 到更小的 $\mathbf{z}={\mathcal{E}} ( \mathbf{x} ) \in\mathbb{R}^{h \times w \times c}$ 。downsampling rate为 $f = H/h=W/w = 2^m$  ，在图像生成最后，通过一个 decoder $\mathcal{D}$ 重新构造图像 $\tilde  x = \mathcal{D}(z)$   
论文提出两种用于autoencoder 训练的正则化方法，避免latent spaces 的高方差问题   

- KL-reg：一个 KL 的penalty，与VAE相近
- VQ-reg：在decoder层加一个vector quantization layer，类似 VQVAE

diffusion process作用于latent vector $z$。这部分的架构是一个time-conditioned U-Net，加上一个cross-attention来获取conditioning information(class, semantic maps...)。每一种类型的conditioning information都有一个对应的 demain-specific encoder $\tau_\theta$ 对应进行编码   

![Pasted image 20250407153258](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407153258.png)  

### Scale up Generation Resolution and Quality

为了生成高像素的图像，一种方法是使用[多个diffusion 构建pipeline](https://arxiv.org/abs/2106.15282)    
![Pasted image 20250407154805](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407154805.png)

在训练过程中加入 noise conditioning augmentation对最终图像的质量至关重要。设输入图像为 $z$，输出图像为 $x$ ，我们需要对 $z$ 进行strong data augmentation再作为下一个diffusion的输入，即$p(x|z)$     
>conditioning noise 减少了compounding error   
>在推理时不需要 conditioning noise

论文发现，最有效的方法是对低像素图像使用 Gaussian noise，对高像素图像使用 Gaussian blur，论文探索了两种数据增强的方法
- Truncated conditioning augmentation: 在低分辨率图像还没完全生成时提前中止
- Non-truncated conditioning augmentation: 完全运行低分辨率的生成，然后添加噪声

另一种方法是 [unCLIP](https://arxiv.org/abs/2204.06125)   
unCLIP高度依赖于CLIP text encoder进行text-guided images生成，给定预训练CLIP和相同数据集训练的diffusion model，我们可以计算 CLIP text embedding $c^t(y)$ 和 image embedding $c^i(x)$   
unCLIP同时学习两个模型  
- prior model $P(c^i|y)$：给定text $y$ 输出 CLIP image embedding $c^i$ 
- decoder $P(x|c^i, [y])$：给定CLIP image embedding 和 y(可选) 生成图像 $x$ 

两个模型生成图像的过程可以写成   

$$
\underbrace{P ( \mathbf{x} | y )=P ( \mathbf{x}, \mathbf{c}^{i} | y )}_{\mathrm{c^{i} \, i s \, d e t e r m i n i s t i c \, g i v e n \, x}}=P ( \mathbf{x} | \mathbf{c}^{i}, y ) P ( \mathbf{c}^{i} | y ) 
$$
这样就与 conditional generation 等价  

![Pasted image 20250407164214](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407164214.png)  

给定一个文本 $y$，CLIP 先生成 text embedding $c^t(y)$，一个diffusion/autoregresssive prior $P(c^i|y)$ 处理这个 CLIP text embedding 得到 image embedding，然后通过一个 diffusion decoder $P(x|c^i, [y])$ 生成图像。  
这样做有几个好处  
- 由于CLIP事先学习了图像和文本的对应关系，所以使用CLIP 的latent space，修改输入文本能够实现 zero-shot image manipulation
- 生成图像时能使用图像作为输入(CLIP进行图像编码)来生成图像的变体，同时保证图像的语义

相比于CLIP，[Imagen](https://arxiv.org/abs/2205.11487) 使用预训练语言模型(frozen-T5-XXL text encoder)来编码文本输入。谷歌的实验指出，更大的语言模型能够有更好的图像生成效果和图像与文本的对齐能力   

![Pasted image 20240709164422](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709164422.png)  

论文发现，当使用classifier-free guidance的时候，增大 $w$ 会导致生成结果更满足文本但是图片效果更差(worse image fidelity)。论文指出这是因为 train-test mismatch。训练时数据 $x$ 在 $(-1,1)$ 之间，但是($w$增大时)测试时生成的图像并非如此。所以论文提出两种thresholding策略  
- Static thresholding：clip x 到 $(-1,1)$ 之间
- Dynamic thresholding：选取一个百分位数的pixel value $s$，如果这个数大于$1$，进行裁剪$(-s,s)$并除以 $s$   

然后就是谷歌常用大规模实验和调参  
![Pasted image 20250407171746](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250407171746.png)   

## Model Architecture

有两种diffusion常用的模型架构: U-Net and Transformer   

### U-Net

[\[1505.04597\] U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)    
Downsampling stack and an upsampling stack   
- Downsampling: 每一步 $3\times 3$ convolutions(unpadded)，加一个 ReLU 和 $2\times 2$ 的max pooling(stride 2)。每一步channels数量翻倍
- Upsampling: up convolution $2\times 2$，加上内部 $3\times 3$ 的ReLU。每一步channels数量减半
- shortcuts: 对应层(down 和 up)间有跳连，为 upsampling process提供high-resolution features

![Pasted image 20250408151647](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250408151647.png)    

另外的，作为U-net的一个补充  
为了能够为图像生成加上额外的条件，[ControlNet](https://arxiv.org/abs/2302.05543)  复制了U-net作为旁路处理额外条件 $c$  
![Pasted image 20250408152249](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250408152249.png)  
ControlNet 做法如下  
1. 冻结原参数 $\theta$
2. 克隆原参数 $\theta_c$ 到旁路
3. 在克隆参数上下加上 zeor convolution($1\times 1$，0初始化)
4. 输出更改为 $y_c = F_\theta(x) + Z_{\theta_{z2}}(F_{\theta_c}(x + Z_{\theta_{z1}}(c))$

>这个模型极其像LoRA(指看上去像)   
>为什么要加入两个0初始化的卷积层，可能的解释是 这样训练从0开始，一开始模型的行为不会有太大变化，也不会太受condition c影响。这样的话有利于模型平稳训练和过渡

### Diffusion Transformer

[DiT](https://arxiv.org/abs/2212.09748)  

DiT是以ViT(Vision Transformer)为基础的模型，谈DiT之前得先谈谈ViT   

首先，CNN具有归纳偏置(Inductive Bias)，在使用CNN的时候，存在以下假设  
- 局部性：图像有意义的模式由局部相邻像素组成(通常如此)
- 平移不变性：特征与位置无关(通常如此)
- 层次结构，空间不变性(pooling)，通道独立性
在这些假设下，有的在多数场景下适用，使得CNN“天然具有先验知识”，在少量样本中容易学习  
而Transformer并没有这样的能力，需要大量数据来学习局部性等信息，虽然适用范围大，但是少量数据表现可能不如CNN类模型   

ViT与上述基本一致，有一系列Transformer+vision的优缺点，其简单实用，成为Transformer用于视觉的代表作   

ViT将图片分为多个patch($16\times 16$)，再将patch投影为固定长度的向量送入Transformer，用与Transformer一样的encoder进行操作。在对图像分类任务中，输入序列尾部会有一个特殊token，对应的输出为最后的类别预测  
![Pasted image 20250408161505](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250408161505.png)   


主要的操作在于 patch embedding  
设定patch大小 $p$，将输入 $I\times I \times C$ 拆分为 $\frac{I\times I}{p\times p}$ 个patch，一个patch是一个输入token。然后加上特殊token cls 加入到 transformer 的encoder中(positional embedding, attention, layer norm, MLP)。  
![Pasted image 20250408162401](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250408162401.png)    

可以不加入特殊token，使用所有token输出的token取平均(average pooling)  

>一些论文的实验结果
>- 位置编码不是很重要，认为是 patch 的位置对模型来说并不难辨认
>- 在小规模数据集上，使用CNN based模型效果更好


DiT 不同的是，它是LDM based 的ViT，处理图像的空间是 latent space   
![Pasted image 20250408183942](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250408183942.png)   
DiT将图像的latent representation和condition，timestep的embedding作为输入，探讨了三种形式   
- In-Context conditioning
	- 将condition和timestep直接作为图像token的一部分进行处理，带来的GFLOPs变化很少
- Cross Attention
	- 将 $t$ 和 $c$ 的 Embedding 连接成一个长度为2的 Sequence，带来的GFLOPs的开销约为15%
- Adaptive Layer Norm(adaLN)
	- $\mathbf{x}^{\prime}=\gamma\cdot{\frac{\mathbf{x}-\mu( \mathbf{x} )} {\sigma( \mathbf{x} )}}+\beta\, \mathrm{w h e r e} \, \mathbf{x}, \mathbf{x}^{\prime} \in\mathbb{R}^{d}$  
	- 其中 $\gamma$，$\beta$ 是通过 $t,c$ 回归得到的

MLP除了回归 $\gamma, \beta$ 之外，还回归 $\alpha$(残差连接的缩放系数)。通过初始化 MLP 使得三个变量初始值都为0，有利于稳定训练  
模型输出noise prediction 和 对角协方差   


## Summary

- Diffusion 是一个理论较为坚实的idea，是相当有意思的想法。Diffusion 兼顾了模型的tractability 和 flexibility，既能够 analytically evaluated 和 cheaply fit data，训练和sample也不会太难。在生成模型领域一定有相当的地位
- Diffusion的推理时间仍然过长(长于GAN)，这有待解决

在现在看来，diffusion的势头略有些衰落，尽管有基于diffusion的新工作出现(例如基于扩散模型的语言模型)，但是没取得足够瞩目的成功   
在GPT-4o的生图能力(大概是自回归)出现后，diffusion 在生图上是否能够与之对抗还有待更多的工作出现。  

>even if you don't believe diffusion models are the future, I don't think you can completely ignore them either and they will probably have at the very least interesting niche applications


## References

Basically notes from Weng Lilian's blog   

- [由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459) 
- [What are Diffusion Models? \| Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 👈 
- [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/) 👈 
- [Fetching Title#ex7y](https://learnopencv.com/denoising-diffusion-probabilistic-models/)
- [Generative Modeling by Estimating Gradients of the Data Distribution \| Yang Song](https://yang-song.net/blog/2021/score/) 👈 
- [扩散模型解读 (一)：DiT 详细解读](https://zhuanlan.zhihu.com/p/685867473)
- [ViT（Vision Transformer）解析](https://zhuanlan.zhihu.com/p/445122996)
