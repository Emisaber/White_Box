---
tags:
  - DL
  - VLM
---


原博客： [Seeing Is Not Reasoning: How VLMs and Their Benchmarks Lean on Text](https://harvey-fin.github.io/seeing-is-not-reasoning/)  
## Introduction

比较意外的出发点  

- 提出code agent 的使用中，如果将高分辨率的图片换成short caption，同时有效的话，可以减少token budget
- 最近的一些工作探索了如果使用caption替代图片，VLM是否同样能够答对
	- 例如PICa(替代也能答对)，Img2LLM(同)，MMStar(流行benchmark中一半是不需要image输入的)，IsoBench(同构image text pair，用text往往答得更好)，MMMU-Pro(去除了MMMU中不需要图像的数据)，MIRAGE(前沿VLM即使在没有图像输入的时候也会产生detailed image descriptions and reasoning traces)

博客通过修改VQA的输入，希望探究，这些benchmark有多少程度依赖于视觉输入，同时VLM在解决问题的时候会不会因为对文本的依赖，在内部形成caption然后进行推理  

## Evaluation Configurations

实验规模是 8 VLMs，9 popular VQA benchmark，5 input conditoins per question

model family   
![Pasted image 20260723165142](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723165142.png)  

benchmark  
![Pasted image 20260723165207](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723165207.png)


input conditions 
![Pasted image 20260723165246](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723165246.png)  

> 注意caption来自于InternVL3-14B，它也是待测试模型  

使用LLM as judge，Qwen3-30B-A3B进行评估  

> 这里提到Rule-based metrics such as ANLS, BLEU, and exact match break down for modern VLMs: models like Qwen3-VL generate long reasoning traces by default, and string-overlap scores reward surface-level wording rather than correctness.
> 大概是说，基于规则的指标因为现在模型的输出会包含大量内容（reasoning），所以即使语义正确也很难被正确评估

## Dimensions of Interpretation

实验的设计是这样的  

- 通过5种condition中的三种noisy input构建一个在没有依赖时的baseline
	- 什么都没有 no images + 白色图像white + 噪声noise
- 然后用带有依赖的输入（区分图像和文本）来评估模型的偏好

$$
g_{\text{m,i}} = \text{acc}(m)_{\text{i}} - \text{baseline}_{\text{i}}
$$
其中i是instance  

基于此定义可以计算三个指标  

##### Modality Differential  

衡量两个模态增益差距  
$$
\text{Modality Differential}_{\text{i}} = \frac{g_{\text{image},\text{i}} - g_{\text{caption},\text{i}}}{|g_{\text{image},\text{i}}| + |g_{\text{caption},\text{i}}|}
$$

数值在$[-1, 1]$之间，数值越大，对image的依赖越强  


##### Caption Substitution

衡量image比caption增益多少，如果没有增益（0），那么图像就是完全可替代的，如果正值，图像就有一定价值  

$$
\text{Caption Substitution} = \frac{\text{acc}(\text{original}) - \text{acc}(\text{caption})}{\text{acc}(\text{original})}
$$


##### Visual Dependence

衡量模型正确回答中需要图像的部分  

$$
\text{Visual Dependence} = \frac{\text{acc}(\text{original}) - \text{acc}(\text{no\_images})}{\text{acc}(\text{original})}
$$

这个指标与其说是需要图像，实际上更接近于需要依赖的部分，至于是依赖图像还是依赖caption就行不一定  

## How Visual are VQA benchmarks

#### Overview

![Pasted image 20260723173039](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723173039.png)  

这个图是8个模型在各个benchmark上指标的average  

- caption substition来说，大部分benchmark 图像和caption没法区分开来
- visual dependence来说，大部分还是需要依赖的
	- MMMU不太需要依赖信息就能直接问答，caption和image也区分不开
	- RealWorldQA竟然没有（图像）依赖答得更好，然后提供图像的增益大于caption


#### caption 和 image 差距


![Pasted image 20260723181848](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723181848.png)  

modality diff指标实际上能够表达上面图像的信息  
数值越大，意味着图像的增益大于caption越多，越依赖于图像，在图1也就是越位于右上方  

#### Low Visual Dependence signals data leakage


![Pasted image 20260723182056](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723182056.png)  

直接通过question就能够答对，甚至输入依赖反而更差，体现了信息泄露


## VLM Capability Evaluation


#### Visual Dependence

![Pasted image 20260723182305](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723182305.png)  

随着模型规模增大，visual dependence下降，可能意味着LLM backbone越强，越能够直接从问题找到答案


#### The image-vs-caption lean is family-specific

![Pasted image 20260723182549](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723182549.png)  

针对模型分析modality diff和caption substitution  
不同family表现不太相同，InternVL系列对语言的依赖更强，Qwen好一些，同时似乎同个family随着规模上升/能力会更依赖于视觉

这里的模型没有“原生”适应视觉的模型，例如K2.5  


#### Case

![Pasted image 20260723182918](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020260723182918.png)  

这里针对MMMU探究了几个模型的指标  

InternVL3.5系列结果比较明显，负值的visual diff表明可以不依赖信息回答问题，caption substitution很低可能表明图像反而对reasoning有干扰

## 总结

文章的总结是  
- benchmark角度：实验证明了有的benchmark是不依赖于视觉的
- 模型角度：对语言的依赖可能不是scale的问题
	- 说 visual dependence 随着scaling 降低
	- 不同的family表现不同
	- 这两点可以支持这个观点
- 然后caption替换可以应用于上下文工程

首先这篇文章的实验设计，指标设计和实验思路应该是可以学习的，这样做确实是比较有价值/比较能够论证假设的做法

不过深究的话实际上做得比较简单，不一定结论成立  

例如说对语言的依赖可能是scale的问题，确实在全局上visual dependence随着scale降低，但是幅度不是很大，可能反而是benchmark自身的问题。在MMMU中，scale和语言依赖没有这样的关系

- 基于InternVL 3.5 14B生成caption，可能caption的正确率受限于模型，如果caption更准确，或许caption的成功率更高
- VLM输入准确描述问题的caption然后答对问题，主要体现的应该是现在的VLM并不喜欢使用视觉/对视觉信息的理解逊色于语言，没能做到真正在视觉上reasoning
- 感觉有点缺失本质，但是不知道少了什么，用语言抽象一层做得更对好像很显然













