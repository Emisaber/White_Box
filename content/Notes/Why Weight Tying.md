---
tags:
  - ML
  - DL
---

### References
- [Weight Tying Explained \| Papers With Code](https://paperswithcode.com/method/weight-tying)
- [知乎-Weight Tying为什么可以](https://zhuanlan.zhihu.com/p/623525701)
- [知乎-碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628) 👈 Recommend  

### Why it works

根据papers with code，有两篇论文独立提出了这个方法   
- [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling \| Papers With Code](https://paperswithcode.com/paper/tying-word-vectors-and-word-classifiers-a)  
- [Using the Output Embedding to Improve Language Models \| Papers With Code](https://paperswithcode.com/paper/using-the-output-embedding-to-improve) ⬅ Transformer中的引用   

我们可以将输入从token到最终输出分为几个阶段   
- tokenizer分词之后得到token id   
- 根据token id 得到 one-hot embedding $h_{onehot}$  `vocab_size, 1`   
- token经过embedding后得到 embedding $h_{in}$  `n_embd, 1`    
- 经过多个Transformer layer得到 相同维度的 $h_{in}$   `n_embd, 1`    
- 经过投影(pre-softmax层)得到  softmax之前的vector $h_{pre}$  `vocab_size, 1`   

Weight tying 的想法在于   
1. 通过token id得到one-hot编码，在 embedding matrix `C, H` 选择对应的embedding，表明**embedding matrix中每一行代表着一个token的embedding**   
$$
h_{in} = U^Th_{onehot}
$$

> 注意embedding matrix这里进行了转置

2. pre-softmax层进行投影时，可以理解为 embedding $h_{in}$ 与 pre-softmax matrix中的每一行计算内积，判断这个 **embedding 与哪一个token(每一行)更相近**
$$
h_{pre} = Vh_{in}
$$

也就是两个矩阵，形状相同，每一行都“代表”一个token的embedding，它们在语义上是相近的   
故而 weight tying 是一定的道理   

> 注意两个层的bias是独立的





