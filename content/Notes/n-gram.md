---
tags:
  - ML
  - DL
---

### What is n-grams

N-grams is the contiguous sequence of $n$ items from a given sample of text or speech   
Types of N-grams: Unigram, Bigram, Trigram, higher-order n-grams    

For example, sentence "The cat has magic power":    
- The Bigrams would be
	- "the cat"
	- "cat has"
	- "has magic"
	- "magic power"

Typically we move one step forward, but you can move multi-step in more advanced scenarios  

When moveing one step forward, the number of n-grams could be calculated by  
$$
N = X - N + 1
$$

where $X$ is the the number of words across the sentence    

### Applications

- **Language Modeling**: Predicting the next word based on the previous $n-1$ words, more details see [[makemore]]  
- **Text Classification**: Identifying categories of text by analyzing word patterns.
- **Machine Translation**: Assisting in translating text by understanding sequences of words.
- **Sentiment Analysis**:  Evaluating opinions expressed in text by examining word combinations

### Advantages & Limitations

- Advantages
	- N-grams help capture local context and word order
	- Improving the accuracy of task like sentiment analysis, tagging and text-to-speech(TTS)
- Limitations
	- As $n$ increases, the dimensionality of the data cna grow significantly, leading to issues such as data sparsity
	- lack of understanding beyond context/window size

