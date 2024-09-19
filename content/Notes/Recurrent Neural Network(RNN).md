---
tags:
  - DL
  - ML
---

## Example Application

### Slot filling

填空  

- 如果使用Feedforward network(前馈神经网络)
	- Input a word
	- 1-of-N encoding
		- 可能需要有other维度来处理词汇表中不存在的表示
		- 对英文单词的话，使用word hashing，例如a-a-a（$26\times 26 \times 26$）
			- apple app, ppl, ple 
	- 作为一个分类任务
		- 每一个词属于各个slot的概率
	- 但是
	- feedforward network不存在上下文信息，所以同一个词的输出不会随着语意和顺序发生改变
		- ![[Pasted image 20240819092816.png]]

## Recurrent Neural Network

### How to keep memory

![[Pasted image 20240819095119.png]]  

增加一个记忆模块，把输出存进记忆里，记忆里的数值参与下一次运算   
大概是这么个流程来实现记忆  

记忆里面的数值随着输入迭代  

![[Pasted image 20240820091110.png]]    
- 仍然是slot filling 的任务，每个word进入得到它的分类情况
- 后一次迭代考虑前一次的内容
- 很明显虽然由后往前有记忆性，对于先输入的词来说，后输入的词没有被考虑到(没有后文信息)

### RNN

#### Elman Network

Elman Network 将hidden layer的输出存到记忆里，参与下一次计算  
![[Pasted image 20240820091612.png]]  

#### Jordan Network

Jordan Network 把最终的输出存进记忆里，参与下一次运算  
传说比较好  

>可以解释为，hidden的输出没有target，比较难学到有用的记忆，最终输出有target，比较好学？

![[Pasted image 20240820091635.png]]    

#### Bidirectional RNN

同时train两个RNN，方向不同，将二者的输出给一个output layer得到最终输出  

这样就能够考虑到前后文  

![[Pasted image 20240820092304.png]]  

### Long Short-term Memory (LSTM)

长 短期 记忆  
一种比较持久的短期记忆机制  
一种记忆的机制，四个部分  

- Memory cell
	- 存储记忆
- Input Gate
	- 是否输入到记忆里
- Output Gate
	- 是否输出参与计算
- Forget Gate
	- 是否忘记当前记忆

![[Pasted image 20240820092834.png]]   

需要有四种输入来控制记忆，产生一个输出  
- 控制三个门的输入
- 可能存入记忆的输入
- 可能产生的输出

#### How does it work

![[Pasted image 20240820093820.png]]
$z$ 表示输入，$a$ 表示输出，$c$ 表示当前的记忆  

Gates的activation function一般是sigmoid，sigmoid输出在0-1之间，代表gate打开的程度  

##### 流程

![[Pasted image 20240820100117.png]]  

1. 输入一个$z$，经过activation function得到$g(z)$， input gate输出 $f(z_i)$相乘，得到输入的新记忆 $g(z)f(z_i)$  
	1. 此时$f(z_i)$作为输入开关，如果$f(z_i)$为0，则输入的新记忆为0
2. $z_f$ 输入经过sigmoid得到 $f(z_f)$，与当前记忆$c$相乘，得到加权的当前记忆 $cf(z_f)$，与输入的新记忆 $g(z)f(z_i)$  加和，更新记忆$c$为$c'$
	1. $$c' = g(z)f(z_i) + cf(z_f)$$
	2. 此时如果$f(z_f)$为0，则旧记忆被遗忘
3. 记忆经过activation function得到 $h(c')$，与output gate输出 $f(z_o)$ 相乘，得到输出 $a$
	1. $$a = h(c')f(z_o)$$
	2. 此时如果$f(z_o)$是0，则没有输出

几个Input都是由模型输入与权重相乘得到的加权和  

##### 结构

LSTM把memory cell 当成一个neuron  
一个neuron需要四组参数  
![[Pasted image 20240820101048.png]]  

对于整个网络  
![[Pasted image 20240820101656.png]]  
- memory cell 变成网络的neuron
- 输入x乘以不同的matrix，得到四组输入向量$z^i,z^f,z^o,z$，dimension与这一层的neuron数量一致

实际上，计算不是以memory cell为单位进行的  

![[Pasted image 20240820102138.png]]  
每个输入向量，$z^i$过 activation function后与$z$相乘，$z^f$ 过activation function后与记忆 $c$ 相乘，二者相加得到新记忆$c'$，新记忆过activation function后与$z^o$过activation function的结果相乘得到output $y$  

然后进入下一轮  
![[Pasted image 20240820102537.png]]  

实际上，输入处还会加入新记忆(peephole)和上一次的输出，把$c, y, x$并在一起  

![[Pasted image 20240820102746.png]]  

然后叠多层  
![[Pasted image 20240820102948.png]]  

现在使用RNN基本就是用LSTM   

> GRU是LSTM的简化版，少了一个gate，参数比较少，据说表现差不多

### Training

如果是slot fliing  
一个序列(句子)作为输入，一个word一个label，对应一个reference vector，然后计算cross entropy  

输入的本质是一个词，但是需要把一个序列当成一个整体，顺序不能改变   

![[Pasted image 20240820104943.png]]  

#### Backpropagation through time(BPTT)

相比于普通的backpropagation多考虑了时间因素  

待补👈 可能补吧  

#### Problem
##### Rough error surface

![[Pasted image 20240820110928.png]]   

error surface 要么很平坦要么很陡峭  
![[Pasted image 20240820111236.png]]  

解决方法可以使用clipping  
设置阈值，当梯度大于阈值时就等于阈值   

但是为什么  
- [[BPTT]]  
- 或者实际例子
- 一个简单的RNN，1层1个neuron
- ![[Pasted image 20240820121202.png]]

###### An Example

输入一个长度为1000的序列   

![[Pasted image 20240820121202.png]]  

通过稍微改变权重，看看梯度怎么变化   

在权重为1和1.01时  **gradient explode**   
![[Pasted image 20240820121628.png]]  

在权重为0.99和0.01时  **gradient vanishing**
![[Pasted image 20240820121748.png]]  

由于序列每个输入之间的联系是累乘的，所以transition部分参数变化的影响是指数级的  

##### Helpful Techniques

- Long Short-term Memory(LSTM)
	- can deal with gradient vanishing 把平坦的部分拿掉
	- 为什么
		- memory和input是相加的
		- forget gate是可以开启的，保留记忆的影响 (一般bias比较小，确保开启)
		- 但是感觉不对👈查一下
	- GRU

- Clockwise RNN
- Structually Constrained Recurrent Network (SCRN)

还有奇怪的点  
- 使用random intialization的话，sigmoid activation function 比较好
- 使用identity matrix 的话，ReLU比较好
	- 一般的RNN就很强





## Applications

- slot filling (vector sequence as input, each word has a label)
- Input a vector sequence, output a vector (多对一)
	- 情感分析
		- 最后把hidden layer拿出来，做一些transform，做分类
	- Key term extraction
- Input a vector sequence, output a vector sequence (多对多)
	- output shorter
		- speech recognition
			- 每个vector有一个label，然后trimming(去重)，有的本来就有重复无法解决
			- Connectionist Temporal Classification (CTC)
				- 允许output null字符
				- 很强
	- 不确定长短 Seq-to-Seq
		- 把原输入给RNN滚过一遍，存下记忆，先产生第一个字
		- 然后第一个字作为输入，结合记忆，产生第二个字，以此类推
		- ![[Pasted image 20240821120912.png|169]]
		- 如何停止 --- 特殊字符
- Beyond Sequence （still sequence to sequence）
	- Syntactic parsing tree 语法分析树
		- ![[Pasted image 20240821121437.png|325]]
	- sequnce to sequence auto encoder
		- 用在文字上
			- 把document 变成 vector
			- 用RNN把句子滚一遍，然后经过decoder得到原句字(类比RNN seq-to-seq)的做法
		- 用在语音上
			- 例如语音比对
			- ![[Pasted image 20240821122357.png|300]]
			- audio segment to vector由RNN来做
				- 声音讯号用RNN滚一遍，最后得到的向量就是我们需要的vector
				- 训练的时候还需要有一个decoder还原声音讯号
				- encoder和decoder一起train
		- chat bot
			- 收集对话
			- encoder-decoder

## Attention-baesd Model

注意力   
Neural turing machine  
input 进来，DNN/RNN控制reading head controller在memory中读取。writting head controller同理。  
![[Pasted image 20240822091822.png]]    

### Related problem

#### Reading Comprehension

![[Pasted image 20240822092308.png]]   

整篇文章转换成语义的vector，query进来，DNN/RNN通过reading head决定哪个vector与reading head有关，重复多次读出信息。   

#### Visual Question Answering

![[Pasted image 20240822092808.png]]  

DNN/RNN 通过reading head选择读取图片的某些位置来获取信息。  

#### Speech Question Answering

![[Pasted image 20240822093035.png]]  
将问题转化成语义的vector，通过语音辨识得到语音的语义。通过reading head选择读取语音语义的部分内容，得出答案  
甚至可以通过attention修正答案，将最终答案和选项计算相似度，得到选项答案   

这些是一起train的  

## RNN & Structured Learning

- RNN LSTM
- HMM,CRF, Structured perception/SVM

![[Pasted image 20240822094044.png]]  

- RNN, LSTM
	- 需要双向的RNN才能考虑整个sequence
	- cost和error不是很相关
	- 但是鬼哉深度学习
- structured learning
	- 能比较简单考虑整个句子
	- 能显式的增加一些限制
	- cost 和 error 是相关的
	- 但是效果不如深度学习(linear)

**二者结合，将deep作为内部，deep的输出作为structured的输入**  

