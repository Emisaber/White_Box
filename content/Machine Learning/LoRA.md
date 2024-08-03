## 杂七杂八的速成尝试
### from [Fine-tuning LLMs with PEFT and LoRA - YouTube](https://www.youtube.com/watch?v=Us5ZFp16PaU)
##### basic idea of PEFT
- Freeze most of the parameters of the pre-trained network
- Not actually training the original weights
- adding some extra weights and  fine tune those
##### advantages:
- avoid catastrophic forgetting （忘记之前的训练）
- 说是不会有这种情况，因为是对另外的参数进行微调

##### parameters

### from [Low-rank Adaption of Large Language Models: Explaining the Key Concepts Behind LoRA - YouTube](https://www.youtube.com/watch?v=dA-NhCtrrVE)

##### fine-tuning
![[Pasted image 20240712230024.png]]  
- 两个权重，正向传播结合输出，反向传播只更新额外的权重
- $h = Wx + \Delta Wx$
##### LoRA
low-rank Adaption  
- pre-trained model have a **very low intrinsic dimension**  
	- 低本征维度，意味着可以用更少的维度来描述
- LoRA提出，the weight also have low intrinsic rank during adaptation
	- 权重如果看成一个矩阵，可能是100 $\times$ 100
	- 但是它的秩可能没有这么多(无线性相关性的列数)
	- 则$$\Delta W = W_A \times W_B$$
	- 其中$\Delta W$ 维度为 $A \times B$
	- $W_A$ 可能为 $A\times r$
	- $W_B$ 可能为 $r\times B$

原计算   
- $$h = W_x + \Delta W_x$$
LoRA
$$h = W_x + W_AW_B$$
- $A\times r$  $r\times B$
- r作为超参数，不一定秩为r，但是至多为r   can't have a rank that exceeds r
- 但是实际上只需要把小的维数设置为r，矩阵的秩本就不会超过小的维数   
![[Pasted image 20240713141803.png]]
这样就可以使用小维度的矩阵来表示权重  
$W_AW_B$的描述的内容和原权重$W$是一样的
推理的时候将额外的权重与pretrained weights结合  

只要有权重矩阵就可以使用这种方式微调  

##### 大概怎么用
- 对于Transformer只对于attention weights
只选择一部分权重矩阵就行(如Transformer $W_q, W_k$)  
选择一个秩 (如$R=4$)
然后训练  
训练之后加载到原权重中(应该是取代)  
- 在多个下游任务，就只需要更换这部分权重就行
- 即插即用

##### 代码的话
- `LoraConfig`
	- 设计


