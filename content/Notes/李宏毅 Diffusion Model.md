---
tags:
  - DL
  - LLM
  - Diffusion
---
## References

- [【生成式AI】Diffusion Model 原理剖析 (1/4) (optional) - YouTube](https://www.youtube.com/watch?v=ifCDXFdeaaM&list=PLJV_el3uVTsNi7PgekEUFsyVllAJXRsP-&index=4)  
- [【生成式AI】Diffusion Model 原理剖析 (2/4) (optional) - YouTube](https://www.youtube.com/watch?v=73qwu77ZsTM&list=PLJV_el3uVTsNi7PgekEUFsyVllAJXRsP-&index=3)
- [【生成式AI】Diffusion Model 原理剖析 (3/4) (optional) - YouTube](https://www.youtube.com/watch?v=m6QchXTx6wA&list=PLJV_el3uVTsNi7PgekEUFsyVllAJXRsP-&index=3)
- [【生成式AI】Diffusion Model 原理剖析 (4/4) (optional) - YouTube](https://www.youtube.com/watch?v=67_M2qP5ssY&list=PLJV_el3uVTsNi7PgekEUFsyVllAJXRsP-&index=2)


## 浅谈Diffusion Model

Denoising Diffusion Probabilistic Model  
DDPM  

#### 如何运作

从Gaussion distribution里sample出一个Vector  
Dimension和目标图片大小一致  

经过一个**Denoise network**，筛去一些杂讯，不断Denoise得到清晰的图片  
denoise的次数是事先定好的，step从大到小  
杂讯到图片的步骤，称为**Reverse process**  



![Pasted image 20240709152149](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709152149.png)  

#### Denoise Model

输入除了杂讯图之外，还有杂讯的严重程度  
这里denoise model一直是同一个  
##### Noise predictor
![Pasted image 20240709152318](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709152318.png)
noise predictor 预测杂讯的样子，然后减掉输入的图片产生输出图片  
![Pasted image 20240709152625](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709152625.png)
- 为什么不直接训练一个end-to-end model直接产生图片
	- 这个可能比较简单
	- 如果产生一个原图加杂讯，那模型基本就能画出原图了，不是很合理
- 有杂讯的图减去杂讯  
- 但是会有很多不同情况，每一步去噪的输入噪声程度都不一样，怎么指明
	- 加入step描述程度
##### Forward process/Diffusion Process
但是在训练时的ground truth怎么来  
人为增加噪声  
对dataset中的图片，人为sample Gaussion distribution中的噪声，一次次加，每一次加带有step数和噪声ground truth  
这个过程称为diffusion process  
#### Text-to-Image
文字生图仍然需要成对的资料  
将文字也作为输入喂给Denoise模组(也就是给noise predictor)  

## 几个流行模型介绍

### 大概架构
三个部分Text encoder, Generation model, Decoder  
分开训练然后组合起来  
##### Text encoder
对文字进行编码，喂给generation model  
##### Generation Model
输入一个杂讯和文字向量，经过中间生成模型(一般是diffusion model)产生一个中间产物  
中间产物是图片压缩的结果  
可以是人类看得懂的也可以是人类看不懂的
##### Decoder
中间产物解码得到图片    

#### Stable Diffusion
SD  
![Pasted image 20240709163518](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709163518.png)  
- 一个encoder，编码各种东西
- 经过diffusion model
- decoder解压缩
#### DALL-E
![Pasted image 20240709163641](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709163641.png)  
- encoder
	- 文本encoder，图像encoder
- Autoregressive 运算量大，生成不完整的图(压缩的版本)
- Diffusion，生成完整的图片
- decoder还原图片  
#### Imagen
![Pasted image 20240709164140](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709164140.png)  
- encoder 编码文本向量
- diffusion model 生成一张不清晰的图
- decoder也是diffusion model，通过几次放大得到清晰图片  

### Encoder
![Pasted image 20240709164422](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240709164422.png)    
- 文字的encoder会有很大的影响
- FID越低越好
- CLIP分数越高越好


两张图表明，文字的encoder比较重要，越大越好，但是diffusion model影响不大  

>[!Frechet Inception Distance]
>FID
>一个pre-trained CNN model, 得到CNN的Latent Representation  
>真实影像的representation和生成的影像的representation  
>假设两组representation是高斯分布(没什么道理)  
>计算Frechet distance，距离越小越好
>![Pasted image 20240710013309](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240710013309.png)  
>需要比较多的sample来进行计算  

>[!Contrastive Language-Image Pre-Training]
>CLIP
>400million的image和text对训练得到的模型
>将文本(描述)和图片编码，相匹配距离越近越好，不相匹配越远越好  
>![Pasted image 20240710014006](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240710014006.png)  
>这样的训练认为，CLIP能判断文字和图片的匹配程度---CLIP score
>- 将文本和生成的图片输入，看结果是否接近

### Decoder

Decoder训练不需要成对的文本和图片  
只需要图片的训练就可以获得这种能力  

- 如果中间产物是小图  
	- 将大图缩小，就得到成对的训练数据
- 如果中间产物是latent Representation  
	- 训练一个Auto-encoder，
	- 图片输入，encoder一下变成latent representation，decode一下，变成原图
	- 训练完后这个decoder就是对应的decoder

>[!Latent Representation大概长什么样]
>人类不可读的一张图
>如果原图$H\times W\times 3$
>latent representation 是$h\times w\times c$
>$h，w$都是$H, W$对应的downsample
>$c$是 channel，表示每一个位置是多少个数字来表示
>也可以把latent representation 看成图  

### Generation Model

diffusion model noise加在图片上，杂讯生图  
现在diffusion model产生的是中间产物，noise应该加在中间产物(小图片或者latent representation)上  

- 怎么得到中间产物
	- 用decoder阶段得到的encoder编码一下
- sample 一些noise不断加到中间产物上

获得数据集之后，训练noise predicter  
- 文字输入
- step
- 对应的加噪声的图片



## Diffusion Model原理剖析
### 第一讲
##### VAE & Diffusion

- VAE是将图片经过encoder变成一个latent representation，然后经过decoder恢复成图片  
- Diffusion有两个过程，forward process 和 reverse process，forward process可以看成是encode过程  

#### 训练算法

伪代码   
![Pasted image 20240710160837](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240710160837.png)  
大概描述   
- 重复以下直到converged(收敛)
- 从数据集中sample一张图出来  <- 原图$x_0$  
- 从1到$T$ 中sample一个数出来    <- 程度指示(步数)
- 从normal distribution 中sample一个 $\epsilon$  <- 杂讯$\epsilon$  
	- 大小和原图一致

然后第五行   
$$\nabla_\theta ||\epsilon - \epsilon_\theta(\sqrt{\bar a_t}x_0 + \sqrt{1-\bar a_t}\ \epsilon, t)||^2\ $$
- 对$x_0$ 和 $\epsilon$ 做一个加权和(weighted sum)
	- 权重$\bar a$是事先定好的，对应数字$(1, ... ,T)$ 
	- $(\bar a_1,\bar a_2, \bar a_3,... \bar a_T)$ 越来越小  
	- 整个$(\sqrt{\bar a_t}x_0 + \sqrt{1-\bar a_t}\ \epsilon)$是一张noisy image  
	- $t$越大$\bar a_t$越小，$(\sqrt{\bar a_t}x_0 + \sqrt{1-\bar a_t}\ \epsilon, t)$ 中 $\epsilon$ 占比越大，代表noise越大
- $\epsilon_\theta$ 代表Noise predictor，将$t$，noisy image传进去
- $\epsilon - \epsilon_\theta(\sqrt{\bar a_t}x_0 + \sqrt{1-\bar a_t}\ \epsilon, t)$，杂讯图减去noise predictor预测的noise
- 也就可以看到，训练的目的就是接近$\epsilon$  
-  为什么是接近 $\epsilon$  
	- 理论上应该是，对特定阶段加入噪音，模型训练来接近这个噪音  
	- 实践上变成，直接sample一个噪声通过权重加到原图上，然后预测原噪声
	- 数学问题

到这里小结一下，DDPM就是理论上，是diffusion process过程不断加噪声，每一次的加的噪声程度逐渐增加，reverse process过程不断去噪，然后实际上通过数学手段，实现变为 指定步数和噪声，按指定的步数权重加噪，然后预测噪声，直接还原图片   

#### 生成过程  

伪代码  
![Pasted image 20240710184628](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240710184628.png)     

大概描述   
- $X_T$ 是从正态分布中sample出来的一个全是杂讯的图
- 遍历T到1， 每次再次从正态分布中sample一个$z$，$z = 0$的时候$t=1$  
- 每次生成下一个图，下标减一
- 直到遍历完

那每次怎么产生下一个图   
$$x_{t-1} = \frac{1}{\sqrt\alpha_t}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_{\theta}(x_t, t)) + \sigma_tz$$   
- ![Pasted image 20240710190256](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240710190256.png)     
- 上一次迭代的图和t输入到noise predicter 中，noise predicter预测杂讯，然后竟然要乘以一个常数得到需要减去的杂讯，原图减去这个杂讯之后，竟然要再乘以一个常数，然后竟然还要再加上一个系数与sample的杂讯z的乘积才得到最终输出   


### 第二讲
#### 影像生成模型本质上的共同目标
从一个简单的地方生成一张图  

在input的地方，有一个简单的distribution(Gaussion Distribution，例如$N(0,1)$)，sample一下，得到一个vector $z$，丢到一个network中$G(z) = x$ 输出一个$x$，$x$是一张图片  
![Pasted image 20240711145023](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240711145023.png)  
- 在高斯分布的空间里，随意取出一个sample，通过一个network都可以产生一张图
- 有network产生的图所构成的空间/distribution，和现实世界的图的空间/distribution 越接近越好

在文生图之中  
- 区别在于加入了对图像的描述(caption)，称condition  
- ![Pasted image 20240711145440](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240711145440.png)  

希望network完成一个映射，从简单的distribution，(加上condition)，映射到复杂的类似于现实世界的图片distribution中，与现实世界越接近越好  

##### 如何才算接近

- **Maximum Likelihood Estimation(最大似然估计)**
- network $\theta$，映射到的distribution $P_{\theta}(x)$，现实世界的distribution$P_{data}(x)$  

**搜集数据集**  $$sample\{x^1, x^2, ...x^m\} \ \ \ from\ \ \ P_{data}(x) $$
**假设能够计算$P_{\theta}(x^i)$** (生成某一张图的几率)  
那么objective function 就会是  
**找到一个使得生成上面sample的图片概率最高的$\theta$**
$$\theta^* = argmax_{\theta}\prod_{i=1}^mP_{\theta}(x^i)$$
- 这个就叫maximum likelihood Estimation
- 最大似然估计
	- 估计一个参数的取值
	- 从样本空间中取样，找到所有可能参数取值下，这些样本发生的事件概率最大的参数


什么道理  
###### 进一步的数学推导

- 取log
$$
\theta^* = argmax_{\theta}\prod_{i=1}^mP_{\theta}(x^i) = argmax_{\theta}\ log\prod_{i=1}^mP_{\theta}(x^i)
$$  
$$
argmax_{\theta}\ log\prod_{i=1}^mP_{\theta}(x^i) = argmax_{\theta}\ \sum_{i=1}^m logP_{\theta}(x^i)
$$
- 求sample出来的$x^i$ 对应的概率和最大，近似于求整个distribution的期望最大
$$
argmax_{\theta}\ \sum_{i=1}^m logP_{\theta}(x^i) \approx argmax_{\theta}E_{x\sim P_{data}}[logP_{\theta}(x)]
$$
- 期望的计算
$$
argmax_{\theta}E_{x\sim P_{data}}[logP_{\theta}(x)] = argmax_{\theta}\int {P_{data}}(x)logP_{\theta}(x)dx
$$
- 也就等价于(减去一个常数)
$$
=argmax_{\theta}\int {P_{data}}(x)logP_{\theta}(x)dx - \int P_{data}(x)logP_{data}(x)dx
$$
- 那就是
$$
argmax_{\theta}\int {P_{data}}(x)(logP_{\theta}(x)-logP_{data}(x))dx
$$
$$
argmax_{\theta}\int {P_{data}}(x)log\frac{P_{\theta}(x)}{P_{data}(x)}dx
$$
- 这项就是$P_{\theta}$和$P_{data}$的[[KL divergence(KL 散度)]]的负值
- 取最大变成取最小
$$
argmin_{\theta}KL(P_{data}||P_{\theta})
$$
- 那就是说
$$
Maximum\ Likelihood \approx Minimize\ KL\ Divergence
$$
极大似然等价于使得两个分布之间距离最小    

#### 怎么算$P_{\theta}(x)$ 

##### 类比VAE的话

![Pasted image 20250316205953](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250316205953.png)  

$$P_{\theta}(x) = \int_z{P(z)P_{\theta}(x|z)dz}$$
> 全概率公式     

$P(z)$ 是高斯分布，但是后一项该如何考虑  
可以这么定义  
![Pasted image 20240711163619](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240711163619.png)  
- 但是大概率基本都是0，很难找到 $z$ 使得$G(z) = x$

VAE中假设，给定 $z$，$G(z)$ 是一个Gaussion Distribution的 **Mean**  
那就有
- 好像是正态分布期望在几何上的性质
- x与期望距离越远，概率在钟形曲线上越小
- 暂时理解为，给定z，模型的输出服从高斯分布，$G(z)$ 是给定z模型理想输出即高斯分布的均值，则此时给定z得到x的概率就与 $G(z)$ 和 x 的距离的相反数成正比 
$$
P_{\theta}(x|z) \propto exp(-||G(z) - x||_2)
$$  

DDPM中，当每一次增加的噪声很小时，可以将 $P(x_{t-1}|x_t)$ 当成是一个高斯分布，我们需要模型预测这个高斯分布，即预测均值和方差。在多数实践中，我们假设模型预测的是分布的均值     
则给定 $x_t$ reverse process 每一步降噪的目标 $x_{t-1}$ 产生的概率和 目标 $x_{t-1}$ 与均值的距离成反比    

$$
\begin{aligned} {{}} & {{} {{} P_{\theta} ( x_{t-1} | x_{t} )}} \\ {{}} & {{} {{} \propto\operatorname{e x p} ( -\| G ( x_{t} )-x_{t-1} \|_{2} )}} \\ \end{aligned} 
$$

有DDPM的 $P_{\theta}(x_0)$ 满足(全概率公式)   

$$
P_{\theta} ( x_{0} )=\int_{x_{1} ; x_{T}} P ( x_{T} ) P_{\theta} ( x_{T-1} | x_{T} ) \ldots P_{\theta} ( x_{t-1} | x_{t} ) \ldots P_{\theta} ( x_{0} | x_{1} ) d x_{1} : x_{T} 
$$  
>所有的T的可能情况下，$P(x_T)P(x_0|x_T)$ 的加和(积分)
>$P_T(x)$  是直接sample出来的，所以和模型参数没有关系


得到这个信息后，我们考虑maximize $P_{\theta}(x)$    
由于过于难算，所以一般转变为maximize $P_{\theta}(x)$ 的 lowerbound  
有如下推导(在[[Variational Autoencoder(VAE)]] 中已经见过一次)   

引入一个简单的分布，等式成立与分布无关  
$$
l o g P_{\theta} ( x )=\int_{z} q ( z | x ) l o g P ( x ) d z 
$$
利用贝叶斯公式写成可拆分的形式，引入一项 $q(z|x)$ 用于化简  

$$
= \int_{z} q ( z | x ) l o g \left( {\frac{P ( z, x )} {P ( z | x )}} \right) d z \ =\int_{z} q ( z | x ) l o g \left( {\frac{P ( z, x )} {q ( z | x )}} {\frac{q ( z | x )} {P ( z | x )}} \right) d z 
$$  
拆分得到  
$$
= \int_{z} q ( z | x ) l o g \left( {\frac{P ( z, x )} {q ( z | x )}} \right) d z+\int_{z} q ( z | x ) l o g \left( {\frac{q ( z | x )} {P ( z | x )}} \right) d z 
$$
发现第二项是KL divergence  

$$
K L \big( q ( z | x ) | | P_{\circ} ( z | x ) \big) 
$$
KL divergence 一定大于0，所以$P_{\theta}(x)$ 一定满足  

$$
\geq\int_{z} q ( z | x ) l o g \left( \frac{P ( z, x )} {q ( z | x )} \right) d z 
$$
这一项可以转化为一个均值   

$$
= \mathrm{E}_{q ( z | x )} [ l o g \left( \frac{P ( x, z )} {q ( z | x )} \right) ] 
$$  
这个均值就是lowerbound    

DPPM 通过类似的推导也能得到一个类似的 lower bound     

$$
\mathrm{E}_{q ( x_{1} : x_{T} | x_{0} )} [ l o g \left( \frac{P ( x_{0} : x_{T} )} {q ( x_{1} : x_{T} | x_{0} )} \right) ] 
$$
- $q(z|x)$ 在VAE中是 encoder，而 $q(x_1:x_T|x_0)$ 是一个给定 $x_0$ 的Diffusion Process
- $q ( x_{1} : x_{T} | x_{0} )=q ( x_{1} | x_{0} ) q ( x_{2} | x_{1} ) \ldots q ( x_{T} | x_{T-1} )$    

### 第三讲

##### $q(x_t|x_{t-1})$ 怎么算

$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \varepsilon
$$

$\varepsilon$ 服从标准正态分布    
则有  
$$
q ( \mathbf{x}_{t} | \mathbf{x}_{t-1} )={\mathcal{N}} ( \mathbf{x}_{t} ; {\sqrt{1-\beta_{t}}} \mathbf{x}_{t-1}, \beta_{t} \mathbf{I} ) \quad q ( \mathbf{x}_{1 : T} | \mathbf{x}_{0} )=\prod_{t=1}^{T} q ( \mathbf{x}_{t} | \mathbf{x}_{t-1} ) 
$$  
所以 $q(x_t|x_{t-1})$ 服从均值为 $\sqrt{1-\beta_t}$ ，方差为 $\beta_tI$的正态分布   

根据 $x_t$ 和 $x_{t-1}$ 的关系，可以直接得到指定步数 t 的分布   $q(x_t|x_0)$  
![Pasted image 20250317211500](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250317211500.png)  
>直接代入，根据正态分布相加得到，详见另一篇笔记 [[Diffusion]]

简化一下，令 $\alpha_t = 1 - \beta_t$，$\bar \alpha_t = \alpha_1 \alpha_2 ... \alpha_t$  

![Pasted image 20250317211718](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250317211718.png)   

给定步数T和超参数 $\alpha_t$，就能直接得到 $q(x_t|x_0)$ 而不用一步步算  

#### 回到 loss function

我们已经得到了 lower bound  
$$
\mathrm{E}_{q ( x_{1} : x_{T} | x_{0} )} [ l o g \left( \frac{P ( x_{0} : x_{T} )} {q ( x_{1} : x_{T} | x_{0} )} \right) ] 
$$

我们可以通过一系列推导得到能够计算的表达式  

$$
\begin{array} {c} {{{{\mathrm{E}_{q ( x_{1} | x_{0} )} [ l o g P ( x_{0} | x_{1} ) ] \ -K L \big( q ( x_{T} | x_{0} ) | | P ( x_{T} ) \big)}}}} \\ {{{{\mathrm{-} \sum_{t=2}^{T} \mathrm{E}_{q ( x_{t} | x_{0} )} \big[ K L \big( q ( x_{t-1} | x_{t}, x_{0} ) | | P ( x_{t-1} | x_{t} ) \big) \big]}}}} \\ \end{array} 
$$

分为三项，第一项和第三项和模型(reverse process)有关，第二项是，diffusion process的过程和从Gaussian sample的概率，和模型参数无关  
说第三项和第一项处理方式接近，只展开第三项的求解   

$$
{{{{\mathrm{-} \sum_{t=2}^{T} \mathrm{E}_{q ( x_{t} | x_{0} )} \big[ K L \big( q ( x_{t-1} | x_{t}, x_{0} ) | | P ( x_{t-1} | x_{t} ) \big) \big]}}}}
$$

这个式子是在 $q(x_t|x_0)$ 分布上，两个不知道什么分布的KL divergence  
分别是 $q(x_{t-1}|x_t, x_0)$  和  $P(x_{t-1}|x_t)$  

怎么计算 $q(x_{t-1}|x_t, x_0)$  ?   
结合乘法公式有这么个过程   

$$
q(x_{t-1}|x_t, x_0)= {\frac{q ( x_{t-1}, x_{t}, x_{0} )} {q ( x_{t}, x_{0} )}} \ ={\frac{q ( x_{t} | x_{t-1} ) q ( x_{t-1} | x_{0} ) q ( x_{0} )} {q ( x_{t} | x_{0} ) q ( x_{0} )}} \ ={\frac{q ( x_{t} | x_{t-1} ) q ( x_{t-1} | x_{0} )} {q ( x_{t} | x_{0} )}} 
$$
>关于中间的式子，$q(x_t|x_{t-1})$ 是怎么来的，我认为是原本应该是 $q(x_t|x_{t-1},x_0)$ 这样后面才有 $q(x_{t-1}, x_0)$ 的展开，但是由于 马尔可夫性质，$x_t$ 与 $x_{t-1}$ 无关，所以直接写成 $q(x_t|x_{t-1})$  

计算出来的这个式子又是什么？  
$$
{\frac{q ( x_{t} | x_{t-1} ) q ( x_{t-1} | x_{0} )} {q ( x_{t} | x_{0} )}} 
$$
经过一番倒推，可以得到这仍然是一个Gaussian distribution  
它的 mean 是 

$$
\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t} x_{0}+\sqrt{\alpha_{t}} ( 1-\bar{\alpha}_{t-1} ) x_{t}} {1-\bar{\alpha}_{t}} 
$$

variance是  

$$
\frac{1-\bar{\alpha}_{t-1}} {1-\bar{\alpha}_{t}} \beta_{t} I 
$$

则 $q(x_{t-1}|x_t, x_0)$ 分布可知   

$P(x_{t-1}|x_t)$  是已知项(就是模型的预测)，也是一个Gaussian distribution     
那就可以计算 KL divergence   

$$
D_{\mathrm{K L}} ( \mathcal{N} ( \boldsymbol{x} ; \boldsymbol{\mu}_{x}, \boldsymbol{\Sigma}_{x} ) \parallel\mathcal{N} ( \boldsymbol{y} ; \boldsymbol{\mu}_{y}, \boldsymbol{\Sigma}_{y} ) )=\frac{1} {2} \left[ \operatorname{l o g} \frac{| \boldsymbol{\Sigma}_{y} |} {| \boldsymbol{\Sigma}_{x} |}-d+\operatorname{t r} ( \boldsymbol{\Sigma}_{y}^{-1} \boldsymbol{\Sigma}_{x} )+( \boldsymbol{\mu}_{y}-\boldsymbol{\mu}_{x} )^{T} \boldsymbol{\Sigma}_{y}^{-1} ( \boldsymbol{\mu}_{y}-\boldsymbol{\mu}_{x} ) \right] 
$$

但我们并不是真的需要这个解析解，我们需要的是minimize KL divergence  
$$
{{{{\mathrm{-} \sum_{t=2}^{T} \mathrm{E}_{q ( x_{t} | x_{0} )} \big[ K L \big( q ( x_{t-1} | x_{t}, x_{0} ) | | P ( x_{t-1} | x_{t} ) \big) \big]}}}}
$$   

现在 $q(x_{t-1}|x_t, x_0)$ 的分布已知，mean和variance都与模型无关，其相当于一个固定的分布。 $P(x_{t-1}|x_t)$ 与模型相关，但是我们只关心它的均值(模型的输出)而不关心它的方差   
所以最小化 KL divergence的方法就是让二者的均值越接近越好   

所以实际上做的就是   
![Pasted image 20250318193220](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250318193220.png)      
给定 $x_t$ 和 $t$ ，模型的输出与 $q(x_{t-1}|x_0)$ 的均值越接近越好   

对右边的式子代入 $x_t$ 表示的 $x_0$，得到更简单的式子   

$$
\frac{1} {\sqrt{\alpha_{t}}} \left( x_{t}-\frac{1-\alpha_{t}} {\sqrt{1-\bar{\alpha}_{t}}} \, \varepsilon\right) 
$$
式子中只有 $\varepsilon$ 是未知的，所以模型需要做的就是预测  $\varepsilon$  

![Pasted image 20250318194207](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250318194207.png)   

回到生成过程的伪代码  
这个 $\varepsilon$ 就是 模型的输出  


#### 小结

实际上的顺序可能是   
在diffusion process 和 reverse process 的前提下   
我们考虑目标函数: maximize $logP(x_0)$  
经过一系列推导，我们发现最终做的相当于使得两个分布$q(x_{t-1}|x_t, x_0)$ 和 模型分布 $P(x_{t-1}|x_t)$ 越接近越好  
在假设下，我们不考虑 $P(x_{t-1}|x_t)$ 的方差，则目标变成 最小化 两个均值的距离   
经过进一步计算我们得到了  
$$
\frac{1} {\sqrt{\alpha_{t}}} \left( x_{t}-\frac{1-\alpha_{t}} {\sqrt{1-\bar{\alpha}_{t}}} \, \varepsilon\right) 
$$  
发现其中 模型真正需要预测的只有 $\epsilon$，则损失函数变成，给定t和图像，得到加入的 $\epsilon$  
然后在生成的时候一步步减去noise   


### 第四讲

但是，为什么最后还有一项  $\sigma_tz$ 呢  
模型的输出被假设为一个Gaussian 的mean，所以为了能够描述一个Gaussian，很自然会加上一个 noise 来还原 Gaussian 的分布  
但是只取 mean 不是也很合理吗   

老师在这里进行了一些猜想   

#### 为什么需要sample

- 有一篇文章在GPT 2上进行了研究，如果只使用概率最大的结果，模型会重复输出相同的话   
	- ![Pasted image 20250318195233](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250318195233.png)  
- 语音合成领域也会在testing的时候加入dropout

所以可能diffusion也是为了避免模型只重复输出概率最大而加入了variance   

#### Application

Diffusion 可以用在语音上   
但是用在文字上比较复杂 (文字是discrete的，难以加noise)  

- 可以加Gaussian在 word embedding上  
- 可以加其它种类的noise

为什么 Diffusion 的结果这么好？  
可能的原因是 将autoregressive的方法加上non auto regressive中   
将一步解决的问题变成多步解决   

有一种 mask-predict 的方法  
Nonauto regressive 一次输出如果举棋不定的话，可以将概率小的盖住，再进行一次decode   
![Pasted image 20250318200417](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250318200417.png)  
>概率大的结果认为是好结果，但是第一次sample的结果可能做得比较糟糕，所以我们再sample一次看看

![Pasted image 20250318200808](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020250318200808.png)    




