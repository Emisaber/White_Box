---
tags:
  - LLM
  - ML
---


>批注或想法以这种方式记录

## From 
[\[1hr Talk\] Intro to Large Language Models - YouTube](https://www.youtube.com/watch?v=zjkBMFhNj_g)
## Intro to LLM
### What is  LLM
#### 2 files   
![Pasted image 20240607000442|300](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240607000442.png)
- the parameters files
- the code to run the parameters
In LLama2-70B, each parameter stored as 2 bytes(float16 number) and so therefore the parameters file is 140,000,000,000 Bytes  

#### Obtain the parameters 

- a chunk of the internet  
- a chunk of GPUs  
- 10TB fo text - 6000GPUs - 12days for llama2-70b 2million dollars  
- generate 140GB parameters.zip file
- zip is a lossless compression, but here is a **lossy compression**  
- 100x 

#### function of parameters
use to predict the next word  
当预测下一个词的时候，实际上需要将大量世界知识压缩到权重中，并不只是token接龙  

**the networks dreams internet documents**  
看上去是正确的，但难以确保哪些是对的哪些是不对的，概率上只是输出符合要求的形式，内容像dreaming  

### How does it work?

**Transformer  structure**  
每一层的数学原理都是可知的  
但是权重分布在整个神经网络中  
每一次只能是整个地调整权重来优化整个模型  

可以将这些权重看成是建立了一个奇怪的不完美的知识库  
- reversal curse
- ![Pasted image 20240608213353|286](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240608213353.png)
- 知识像是一维的，而不是确切地存有知识，能从各个方向访问，只能是单向
- **empirical 经验主义**  

>empirical 经验主义让我想起耳熟能详的哲学话题 ”**经验论和唯理论**“的争辩
>- 大模型生成的能力，更多体现着一种经验主义，而对大模型 的增强有往其中加入符号，理论原则的趋势
>- CoT，ToT，BOT等方法，实际上就是创建了一套生成(行动)的理论原则，遵循原则之下模型表现确实得到增强。
>- 纯粹经验很明显比较弱，建立行之有效的原则，或者说糅合符号智能未尝不是道路，大概  
>- ~~我在胡扯什么~~
>顺便地，哲学上的唯理论和经验论之争由康德结束，康德在这里的理论还没有学。


### fine tuning
Internet document generators阶段，称预训练，第一阶段   
fine tuning 如果有的话，为第二阶段  

**obtain assistant model**  
- keep the trainning the same -- next word prediction
- modify the data set
	- 一般是手工收集数据
	- 为特定的数据打标签
	- ![Pasted image 20240608215345|272](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240608215345.png)
	- 好像，训练语料的形式就是对话式的(因为有这个目的？)，这个形式和调用API时输入的prompt部分很相似，应该就是因为以类似的方式进行训练？
微调后，模型使用assistant的方式输出，但仍然能使用前一阶段的训练得到的知识，同样难以被解释  

### Procedure


![Pasted image 20240609003034|400](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609003034.png)

**第一阶段称为pre-trainning, about knowledge**  
**第二阶段称为fine-tuning, about alignment**  

>如果第二阶段只是学习这样的服务格式的话，基座模型的能力基本就决定了模型表现，所以微调不是很能够改变模型专有能力？**应该不对**


所以大模型大概是这么生产的  
1. 预训练模型(几个月)
2. 微调(一两天)
3. 测试，找出问题
4. 解决这些问题


**怎么解决问题**  
- 大概地讲
- 微调后得到的模型出错的地方，人为修正，整合到数据集然后再train
- 迭代过程


### Second kind of label

In fine-tuning stage, step 2 and 3  
**comparison labels**  
比较哪个label好要比直接写答案简单  
![Pasted image 20240609003341|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609003341.png)
通过对比模型输出，选择一个最好的加入训练集  
openai称为：reinforcement learning from human feedback  通过人类反馈来进行强化学习  

 **现实中label逐渐变成人和模型协同的工作**  

## Improvement

#### scaling laws:  
- **Performance of LLMs is a smooth, well-behaved, predictable function**
- with 2 varibles:
	- **N - the number of parameters in the network**  参数量
	- **D - the amount of text we train on**  训练的文本量
**only these two numbers can predict what accuracy LLM will achieve  on the next word prediction task**  
and the trend **do not show signs of topping out**  
不会到达顶点  
algorithm is not that necessary(as nice bonus), but we can just get better model for free with better computer  

> 力大砖飞！

![Pasted image 20240609005827|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609005827.png)

#### evolving
以chatGPT为例  
实际上使用时不是完全由模型生成，会使用工具  
- web
- 计算器
- 代码

using tools and infrastructures, tying everything together  

>藏在Web背后的大模型基本就是一个工具集合体，agent的路是不是已经被大企业们遥遥领先

## Future

#### System 2
![Pasted image 20240609013341|400](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609013341.png)

可以认为，人类有两种思考方式  
- 一种是system1，自动的，快速的，无需过多思考的，感情化的，直觉的等等
	- speed chess
- 一种是system2，理性，慢
	- competition

现在的大模型基本是第一种  

使用时间换取准确度  
ToT  

#### self-improvement

- alpha go
- 人类的比赛，模拟，无法超过人类
- 自己和自己比赛 —— 强化学习

对于大模型来说，难以确认奖励和惩罚  
**在小的领域内可行**   

#### Custom 
#### LLM OS
![Pasted image 20240609014639|425](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609014639.png)
as kernel of a emerging operating system  

将现有的都合起来  
- 对话能力
- RAG个人语料检索
- 多工具使用
- 多模态，语音，图片，视频
- 微调形式上的优化
- 个性化设计
- 小领域的self-improvement

multi-threading, multi-processing, speculative execution(预测执行), user space, kernel space 都能在LLM OS中找到对应抽象   

除此之外：
- 主流系统与其他系统构成的庞大生态  
- ![Pasted image 20240609015613](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609015613.png)


## Security

#### jailbreak attacks
越狱攻击  
![Pasted image 20240609020024|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609020024.png)
![Pasted image 20240609020107|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609020107.png)

- base64 encoding of the same query
- ![Pasted image 20240609020224|300](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609020224.png)
- 可以认为是另一种语言，一些语言也可能产生问题
	- 安全问题大多基于英语
	- 没有学会拒绝非安全query，只是学会了某种模式而已

- universal transferable suffix
- 训练出来的攻击模型的后缀

- 噪声处理(训练)出来的图片实现越狱

#### Prompt injection
![Pasted image 20240609021048|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609021048.png)

将prompt藏在人眼不可见的位置，当LLM看到这段prompt时，会忘记之前的prompt（取代原来的prompt），执行当前prompt

- 将prompt injection attack放在网页里
- ![Pasted image 20240609021327|325](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609021327.png)

- 放在文档里
- ![Pasted image 20240609021657|350](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609021657.png)
- 谷歌通过图像的访问权限防止了这个攻击
- 但是又有工具能实现这个权限

#### Data poisoning/Backdoor attacks


**trigger phrase** trige LLM to perform specific prediction  

![Pasted image 20240609022325|450](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609022325.png)

trigger words corrupt model  


![Pasted image 20240609022518|375](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240609022518.png)


