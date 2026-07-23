---
tags:
  - DL
  - Generative
---


[Generative modelling in latent space – Sander Dieleman](https://sander.ai/2025/04/15/latents.html)  

试着看看博客，一方面理解一下什么是latents，一方面看看25年4月左右，大概有什么样的相关知识  

博客中，latents代指 latent representation  
然后这篇博客关注 generative model 语境下的latent space（看完再回来看这个表达是否正确）
## The recipe

现有generative model训练通常分成两个阶段  
- 训练autoencoder on the input signals，encoder + decoder
	- encoder将input signal map为对应的latent representation，decoder map回input domain
- 在latent representations上训练 generative model
	- AR或diffusion

一般第一阶段之后，encoder会冻结，latent space下的generative model第二阶段的训练目标通常和decoder无关（不过现在似乎有了pixel space，就不一定了）

![Pasted image 20260721161303](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260721161303.png)  


为了训练映射关系+高保真，一般encoder + decoder的训练会涉及多个loss function，可能包括一个regression loss，一个preceptual loss，一个adversarial loss  
为了限制capacity of the latents，可能会加入额外的bottleneck loss  

- regression loss通常包括 MAE(mean absolute error)，MSE
	- 只有重建损失，为了更低的损失可能会导致模糊化，失去高频信息
- preceptual loss有多种形式，一般是用另一个frozen pre-trained neural network来提取perceptual features，loss希望让重建的结果和input之间，这些feature是相近的，来更好地保留高频信息（better prevervation of high-frequency content that is largely ignored by the regression loss）。常见的loss有LPIPS
- adversarial loss，训练一个额外的discriminator，鼓励生成结果的真实性realism，但是它不约束一定和输入是一致的（不是重建损失），所以可能会导致输出偏离输入
	- 更像真实的输入分布，但是位置可能又不准
- bottleneck loss用于规范latent空间，直接进行训练有可能会出现，latent直接把像素编码进去，或者训成不包含核心语义信息的表达（只对重建有效），为了限制其内容需要加入bottleneck
	- 例如VAE的bottleneck loss是KL散度，要求encoder输出符合高斯先验
	- VQ-VAE要求encoder输出靠近某个code book中的离散向量

>[!notes]
>#### 高频和低频信息
>低频信息是指变化缓慢的信息：整体亮度，大面积平滑区域，大致轮廓，结构等
>高频信息是指局部快速变化的信息：例如边缘，纹理，毛发等
>重建损失往往会导致平滑结果，对抗损失和perceptual loss能够帮助恢复高频信息
>对抗损失为了更像原分布，比起平滑后的结果更喜欢原结果（实际上可能是discriminator的判断原理是通过卷积等网络特征提取导致的）；而perceptual loss本身就特征对比，所以有帮助


## How we got here

无论是AR还是diffusion，一开始的工作都是在raw digital representations of perveptual signals（pixel或者waveforms）上设计的  

但是很快发现，**perceptual signals mostly consist of imperceptible noise**，或者说尽管存在大量的信息，只有少部分信息是有效的（低维流形）

所以转向了latent space  

#### Latent autoregression

VQ-VAE开创性地提出了学习离散的representations：先通过卷积层downsampling，然后再经过code book/bottleneck layer离散化  

之前autoregression的一次一个pixel现在变成了一次一个latents，极大地加速同时不会引入太多imperceptible noise

VQ-VAE得到的是一个带空间结构的latent（VAE是一个无结构的向量）
$$
z_e(x)\in\mathbb R^{h\times w\times d}.
$$

对每个位置单独量化，得到离散化网络，每个位置仍然大致对应真实图像的对应位置，保留了对应的结构信息  
这样的spatial structure让当时的pixel-based model很容易适配  

VQ-VAE2进一步提高了分辨率到256x256，VQGAN引入了adversarila learning 

> sander认为VQGAN可能是GAN获得时间检验奖的原因，它引入的损失形式就是重建 + perceptual + 对抗 + bottleneck，这一套至少保留到了现在


#### Latent diffusion

2020s左右，很多工作几乎同一时间在diffusion上也进行latent的尝试，比较有代表性的是[\[2112.10752\] High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 他们复用了他们之前VQGAN的工作，将自会归transformer改成UNet，形成了stable diffusion 的基础


之前的一些工作的结构是  先pixel level生成低分辨率的图像，再经过upsample diffusion model得到高分辨率图像，例如DALL-E 2，Imagen 2  
Stable Diffusion 之后大部分都改成在latent上进行

对于AR类模型来说，训练只需要最大化likelihood，而对于diffusion来说，损失是对所有noise level（对应的损失）取期望，所以训练的时候对不同level的noise施以多大的权重很重要

这使得diffusion loss有一点perceptual loss的感觉，我们通过调整noise的权重，实际上是在控制应该学习到那些perceptually matter的信息。而如果可以直接通过diffusion来学习到这一点，似乎两个阶段的训练有些多余

博客提到实际上二者是互补的  

- 感知在小尺度和大尺度上的处理似乎是有很大区别的（特别是视觉），对于texture和fine-grained detail值得单独处理。而adversarial适合这个任务
- 使用compact latent space对算力要求更小

## Why two stage


- 为什么需要encoder阶段来确认latent space
	- 现有的一些lossy compression可能可以在重建上work，但是他们不适合于model/ latent 更容易model
		- 这里有一个reconstruction quality and modelability的trade off
	- latent 保留了一些structure信息，只要有合适的inductive biases就可以被generative model利用起来
	- 另一个点是：latent representation能够更好地表达 perception works differently at different scales这件事
		- 对于audio，人类对高频变化比对低频变化可能更敏感
		- 对于视觉，局部的快速变化和全局信息可以区分textures和structure (stuff vs. things)

这里给了一个经典例子区分 texture 和 structure （stuff或things）

对于一张草地上的狗的图片  
![Pasted image 20260721211743](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260721211743.png)  

我们对草地的感知是一整片field，认知为草地，不会关注每根草的细微变化，数量等信息，但是如果仔细看草地，草的变化是高频的：the grass texture(stuff) is high-entropy  

而对于狗的眼睛（structure/things），一旦出现变化，我们很容易察觉

说  A good latent representation will make abstraction of texture, but try to preserve structure  

使用latent而不是pixel level，更顺应/更能够表达这种差异，latent可以自然地丢弃对texture的详尽描述，只保留必要的texture，同时保留足够完整的structure  


## Trading off reconstruction quality and modelability

大部分lossy compression algorithm基于rate-distortion theory  
> 衡量我们能够进行压缩的程度(rate)和我们允许压缩的程度(distortion)之间的关系

使用latent representation同样可以进行lossy compression，同时会引入第三个因素：modelability（how challenging it is for generative models to capture the distribution），形成一个三因素的trade off

需要modelability的主要原因，就是generative的时候，发现是需要structure信息的，lossy compression往往不包含structure信息，因为它作为先验的一部分，会直接从decoder恢复，所以lossy compression的方式基本不可行

更好的representation应该能够behaving like pixel，但是又不至于冗余  

![Pasted image 20260721221541](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260721221541.png)  


## Controlling capacity

对于容量来说，downsampling factor和number of channels of the representation很重要  
也就是构成hard bottleneck的设计，对于离散的latent，自然就是codebook size  

downsampling是相对于pixel(input) width 和 length的限制，number of channels则是限制长宽之后对单个vector信息量的限制，在这里的讨论下，通常downsample之后信息量少于输入

但是在RAE下，图像虽然仍然downsample，但是会改成高维向量，不一定信息容量减少  

![Pasted image 20260722003749](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260722003749.png)  

> 信息容量/数值个数减少的倍数可以称为 tensor size reduction factor(TSR)


如果TSR不变，改变其它的因素，可能结果不会差别太大，如果TSR改变，通常会对结果产生很大影响


> 博客提到，按理说尺度和channel的设置不会影响信息量，因为单个数字的信息数学上可以是无限的，但是实际不是，有以下可能原因
> - 实际数值精度有限
> - 很多实现里encoder会加入noise，产生影响
> - 当前的神经网络不擅长学习highly nonlinear function


> 博客还提到，神经网络更倾向于学习简单的function，同时如果真的能够实现highly nolinear mapping，会影响modelability


## Curating and shaping the latent space

capacity是信息容量问题，还需要解决
什么信息应该被保留，这个信息怎么被表达  
称前者有curating the latent space，后者为shaping the latent space  

#### VQGAN and KL-regularised latents

来自VQGAN，有两种做法  
- einterpreting the quantisation step as part of the decoder
- Remove the quantisation step from the VQGAN recipe altogether, and replace it with a KL penalty

把离散化放在decoder，虽然可以在最后建立bottleneck，但是很多时候encoder才是信息限制的主要位置  

第二种做法KL，是引入ELBO的一部分，鼓励latent follow imposed prior distribution  

ELBO的实际形式是  
$$
\log p(x)
\geq
\mathbb E_q[\log p(x\mid z)]
-
D_{\mathrm{KL}}(q(z\mid x)\|p(z)).
$$

但是实际使用的时候，KL项的权重会被设置得很小，损失优化不能再认为是最小化negative log likelihood的上界（不再是严格的最大似然变分学习），KL项更接近于一个约束latent的正则项  

$$
\mathcal L_\beta
=
-\mathbb E_q[\log p_\theta(x\mid z)]
+
\beta
D_{\mathrm{KL}}
\big(q_\phi(z\mid x)\|p(z)\big),
\qquad \beta\ll 1.
$$

> 这里的原因是，KL项约束太强，影响了重建性能
> 同时，前面提到了这个约束项让输出接近Gaussian distribution，但是因为weight很小，实际上也没有这个效果，所以encoder输出确实不会是Gaussian  

#### Tweaking reconstruction losses

reconstructuion loss倾向于保留low-frequency content（如果没有其它损失，重建结果会倾向于blurry）。这个结果并不是因为有什么数学倾向，单纯是因为图片中的low-frequency的信息更多  

博客提到，对于一张natural image，the power of different spatial frequencies tends to be proportional to their inverse square  
$$
P(f)\propto \frac{1}{f^2},
$$
> 该频率附近信号成分的平均功率和频率平方倒数成正比
> 大概理解就是，越高频占比越小

所以重建损失需要其它损失辅助，那为什么不能直接去掉有瑕疵的重建损失？

博客提到其它两种损失难以优化，同时往往有pathological local minima  
重建损失可以作为一个约束项来指导参数空间的优化  

> pathological local minima：病态局部极小值，指的是优化算法找到了一个附近无法继续明显降低损失的位置，损失数值可能不高，但对应的实际输出非常差，甚至利用了损失函数的漏洞。

有一些对重建损失的优化工作，等到需要的时候再来参考


#### Representation learning vs. reconstruction

根据上面的讨论，reconstruction loss不只保证了重建质量，同时约束了representation中应该保留的信息（curating reprsentation，起到重要作用），它负责了这两个任务，sander认为它并没有真的做得很好


训练一个encoder得到合适的表征，和得到高质量的重建，实际上是两个任务，现有autoencoder的做法实际上简化的：将两个任务一起训练，作为generative model的第一阶段


sander自己团队的工作尝试着引入辅助decoder来负责curating，主decoder专注于重建（主decoder的梯度不会传回encoder，不对latents产生影响），虽然这个工作影响力有限，但是sander argued 这样的想法仍然是比较有价值的


#### Regularising for modelability

- Capacity决定了信息量
	- 信息量越多，generative model得到的信息越充分，效果可能越好
- Shaping主要是效率问题
	- 相同的信息可以被不同的方式表达，选择更容易的方式会更高效
- Curation决定保留的信息
	- 如果任由模型选择，可能会编码更容易model的噪声，影响性能

> [Rudy Gilman on X: "The sdxl-VAE models a substantial amount of noise. Things we can't even see. It meticulously encodes the noise, uses precious bottleneck capacity to store it, then faithfully reconstructs it in the decoder. I grabbed what I thought was a simple black vector circle on a white https://t.co/eK7ZtLJ6lc" / X](https://x.com/rgilman33/status/1911712029443862938?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1911712029443862938%7Ctwgr%5E3b079097197268a5ac9b65e523037b472ec58836%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fsander.ai%2F2025%2F04%2F15%2Flatents.html) 一个SDVAE噪声的例子


sander这里引入了$\mathcal{V}$-infomation，$\mathcal{V}$-infomation 衡量的是 信息的可用性，对于observer来说，一部分信息在计算上有多大的挑战性  

> the usability of information varies depending on how computationally challenging it is for an observer to discern

对于generative 模块来说，如果latent中的一部分信息需要复杂的架构来提取，那么它的$\mathcal{V}$-information就会很低。maximize latents 的$\mathcal{V}$-information是很有必要的

一些可能的方式是  
- 使用generative priors
	- 在训练autoencoder的时候，同时训练一个lightweght的 generative model来在latents中引入这样的先验
- 使用pre-trained representation
	- RAE
- 鼓励equivariance
	- equivariance是等变性，先变换图像再进行编码得到的结果接近于先编码再变换latent的结果，这样的性质可以让latent在空间上更平滑（不会出现无意义的跳变）

#### Diffusion all the way down

有另外的一个分支用diffusion作为decoder   

这个分支的观点是  
- 通过diffusion decoder训练得到的latents，provide a more principled, theoretically grounded way of doing hierarchical generative modelling
	- 更原则化和理论基础指的应该是diffusion本身理论就是比较严格的条件生成模型，把latents + reconstruction层次化区分得更好，相比起VAE多损失叠加也更有可解释性
- 只需要MSE loss，更鲁棒
- 使用iterative refinement decode可以提高输出质量
	- 这里的iterative refinement，从语境来看不是在latent space，所以这个做法显然有成本问题
	- 但是实际上可以用distillation的方法来加速diffusion

diffusion autoencoders的最大优势应该就是损失的简化，不需要adversarial training对训练来说是比较理想的，如果能够在不使用adversarial loss同时达到相近的效果，又缓解/解决latency问题，可能就可以替代当前的autoencoder


## The tyranny of the grid

Digital representation of perceptual modalities通常是网格结构的  
他们是现实世界的underlying physical signals的均匀采样  

> 按照我的理解，指的是，现实世界是连续的，无论以何种形式获取电子/数字形式的信息，得到的都是离散化后的，通常又是网格化的
> pixel本身就是网格的，video，audio只是网格形状的不同


网格化的处理一方面有效表达了现实世界的信息：perceptual signals 通常是在时间和空间上是静止的，网格没有破坏这种拓扑结构，方便我们学习这些信息，同时也方便设计网络结构

目前的模型基本都是以网格化为基础设计的

但是现实世界的perceptual signals是不均匀的，相同大小的网格包含的信息是不均匀的，而现在处理时又是等价处理，势必存在很多waste和redundancy

![Pasted image 20260722151906](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260722151906.png)  



但是如果直接使用非网格化的结构，对硬件和模型设计都很不友好，目前没有太多的发展，大多是允许这个冗余存在  

sander指出，transformer通常被认为是序列模型，但是它进行的是集合的处理，大概的意思应该是，CNN，RNN的对拓扑的设计是包含在结构内的，transformer对输入拓扑的感知来自于位置编码等外部信息，所以会更灵活，支持更多形式的（例如variable-rate的输入）


有一些工作会放宽topology of the latent space，这些工作也往往得到了更加semantically high-level的representations


## Latents for other modalities


image 的latents已经是被较好探索了的领域  

- 对于video
	- spatiotemporal latent representation仍然存在很大的探索空间
	- 如何考虑人类对motion的perception等仍然有待探索
- audio
	- 即使已经一定程度使用了image的范式，但是怎么让它更work仍然不太清楚
- langauge
	- 语言的冗余程度远小于image，例如shannon评估英语存在50%的冗余，而图像的冗余是量级的差距
	- 通常tokenizer使用的是无损的压缩

## Will end-to-end win in the end?


这里讨论的是是否应该转向端到端（而不是两个阶段，甚至不是使用latents）

在我阅读的现在有很多工作确实强调端到端，目前似乎也没有真的主流或者证明足以取代主流generative model范式

sander认为转向端到端现在还太早了
看起来的主要原因有   
- latent带来的efficiency似乎还不能被直接放弃
- 目前还没有能够scale起来的替代方案

