---
tags:
  - ML
  - DL
  - LLM
---

Tokenizer的作用是将输入的自然语言转化为机器可读的数字序列。   
这通常包含以下几个过程：   
- Input Text: natural language `Fine Tuning is fun for all!`
- Tokenization: breaks down the text into pieces called token(word, subword, character...)
	- `["Fine", "Tuning", "is", "fun", "for", "all", "!"]` word
	- `["Fine", "Tun", "##ing", "is", "fun", "for", "all", "!"]` subword `##` indicates `ing` is part of the previous word
- Token IDs: Each token is then mapped to a unique ID based on a vocabulary file that the tokenizer uses (token 与 整数/字典索引的映射，取决于tokenizer使用的字典)
- Encoding: convert token into token id
- Decoding: convert token id into token/human-readable text

tokenizer实际上决定了模型理解文本的起始形式，它对语言的encoding可能会对模型能力产生影响，特别是在数字的处理上    

一个简单的例子(来自 [Demystifying Tokenization: Preparing Data for Large Language Models (LLMs)](https://www.linkedin.com/pulse/demystifying-tokenization-preparing-data-large-models-rany-2nebc/))    
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

text = "My name is Rany ElHousieny."

encoded_text = tokenizer(text)["input_ids"]

print(encoded_text) 
```

```
[3220,1416, 310,416,1279,3599,41,528,1914,90,15]
```

## Padding & Truncation

### Padding
Ensure that all sequences in a batch are of the same length   

Padding是在模型通过batch input进行训练或推理需要进行的数据处理，最主要的原因在于batch/tensor要求输入序列在各个维度上(在这里是时间维度)是一致的(B, T, C)    
一致才能进行一系列相关的矩阵运算    
> 如果输入只有一个序列(batch=1)，并不需要进行padding   

padding通常使用一个特殊token作为标识   
在多数tokenization系统中，pad token是0，有的系统会加上一个attention mask标识哪些token是需要关注的    

`0` as pad token   
```
[[23, 567, 234, 53, 12, 3456], 
 [45, 678, 0, 0, 0, 0]]
```

with attention mask, `32000` as pad token    
```
{'input_ids': tensor([[1, 887, 526, 451, 263, 13563, 7451, 29889],
                      [1, 887, 526, 451, 263, 13563, 7451, 29889],
                      [1, 887, 526, 451, 29889, 32000, 32000, 32000]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0, 0]])}
```

但是，由于GPU资源宝贵，使用padding的话一次输入只能训练一个样本，通常在训练的时候不会padding而是选择packing：将多个样本拼接(达到目标长度)作为一个输入，并通过attention mask指定需要考虑的样本    

#### 关于left padding 和 right padding

调查的时候没有发现很权威的说法    
比较统一的有以下几点    
- 一般训练的时候使用packing，推理的时候使用left padding
- 推理使用left padding的原因是，如果使用right padding，pad token可能会影响输出结果: 第一个next token变成pad token的next而不是原文本的next，除非指定从哪里开始推理(Llama2原仓库的实现似乎就是找到最短的一个batch，从第一个next位置开始推理)  

一些参考资料  
- [Can GPT2LMHeadModel do batch inference with variable sentence lengths? · Issue #3021 · huggingface/transformers](https://github.com/huggingface/transformers/issues/3021#issuecomment-1232149031)
- [LLM的pad策略，为啥训练时是right，预测是left？](https://www.zhihu.com/question/6455911112)   
- [The effect of padding\_side - 🤗Transformers - Hugging Face Forums](https://discuss.huggingface.co/t/the-effect-of-padding-side/67188/5)
- [LLM padding 细节](https://zhuanlan.zhihu.com/p/675273498)

hugging face库的实现一般会用EOS token 作为pad token   

### Truncation

当输入超出模型限制时，往往需要进行truncation(截断)    
需要注意的是   
- truncation可能会带来信息的丢失
- truncation可能会改变输入的含义或分布
- 可以采取一定的策略进行truncation，例如：保证关键信息不被截断，重写总结前文....

## References

- [Demystifying Tokenization: Preparing Data for Large Language Models (LLMs)](https://www.linkedin.com/pulse/demystifying-tokenization-preparing-data-large-models-rany-2nebc/)