---
tags:
  - ML
  - DL
---


### Overview

For short, MinHash is a technique used to estimating the similarity between two sets.  
It was introduced by Andrei Broder in 1997, primarily to aid in detecting duplicate web pages   

MinHash operates on the priciple of locality-sensitive hashing(LSH). The fundamental idea is to represent a set by its minimum hash value derived from multiple hash function, avoiding access the details of all elements.   

#### How it works

- **Hash Functions**: select $k$ different hash function, and generate hash value of all element in each set $S$   
- **Minimum Hash Value**: For each hash functions, the minimum value obtained from the hashed elements is taken as a representative value for that Set
- **Similarity Estimation**: Counts how many times the minimum values of both Sets(A, B) are equal(across k functions). The similarity can be calculated as:
$$
 J(A, B) \approx \frac{y}{x}
$$


> The similarity here is the estimation of Jaccard similarity, which defined as  
>   $$
\operatorname{J a c c a r d} ( A, B )={\frac{| A \cap B |} {| A \cup B |}} 
$$



### Implement

#### An easy version

[MinHash — datasketch 1.6.5 documentation](https://ekzhu.com/datasketch/minhash.html)  
**datasketch** lib offer a MinHash class  

```python
from datasketch import MinHash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

m1, m2 = MinHash(), MinHash()
for d in data1:
    m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and data2 is", actual_jaccard)
```


#### Maybe I will

Implement by myself  

- [Minhash LSH Implementation Walkthrough: Deduplication](https://dzone.com/articles/minhash-lsh-implementation-walkthrough)
- [Fast Document Similarity in Python (MinHashLSH) - Codemotion Magazine](https://www.codemotion.com/magazine/backend/fast-document-similarity-in-python-minhashlsh/)

Just for fun or learning, it may not efficient enough compare to the easy one?  

