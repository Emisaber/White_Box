---
tags:
  - DL
  - ML
---

# Image Classification  

假设图片大小相同  
![[Pasted image 20240816083203.png]]  

## 对图片的处理  
![[Pasted image 20240816083711.png]]  

1. 当成三维的tensor  
$$(H, W, C)$$  
- 高(H)，宽(W)，channel(C)  
- 高与宽代表pixel，图片的大小，channel是RGB  
- 每个数值是RGB的强度

2. 拉长为向量，向量长度为$H\times W \times C$

## Model
### 一种思路

if Fully Connected Network  
![[Pasted image 20240816084230.png]]  
- 参数量过大
- 尽管模型弹性变大，但是容易overfitting，而且不好练

#### Obeservation 1
每个神经元并不需要看到整张图片，只需要看到其中一部分(pattern)就可以  
![[Pasted image 20240816084922.png]]  

##### Simplication 1
设定一个**Receptive field**(感受野)  
也就是在3-D tensor中圈定一个范围  
一个神经元只考虑自己的Receptive field  
![[Pasted image 20240816085536.png]]  

##### 如何确定Receptive field  

自己决定(domain knowledge)  
- 同一个范围可以多个receptive field，  
- receptive field之间可以重叠  
- 不同神经元使用不同大小的receptive field  
- 不同神经元使用不同的channel
- 非正方形的receptive field
###### Typical Setting 1

- 考虑全部的channel  
- receptive field的大小称为 **kernel size**，不会太大，一般$3\times 3$  
	- 那pattern一定能在$3\times 3$的范围内侦测出来吗 [[#^3eb341|关于kernel]]  
- 一个receptive field有一组neurons（例如64个）
- receptive field经过一个步幅(称**stride**，一般1或2)，垂直与水平移动，产生下一个receptive field
	- 在边缘时做或不做padding，有多种padding方法  

![[Pasted image 20240816092709.png]]  

#### Observation 2

同一个pattern可能出现在图片的不同地方  
在上面的简化中，receptive field覆盖了所有地方，即使不同的地方也会被检测到   
实际上这几个神经元作用是一样的，为了简化，考虑共享参数  
##### Simplication 2

![[Pasted image 20240816093306.png]]  

功能一样的neuron参数一样  
> 他们的参数一样，但是输入不一样，输出就不一样
> 同一个receptive field的neuron如果参数一样，输出也会一样，所以一般不这么做

###### Typical setting 2

以receptive field为单位，所有的receptive field的多个neuron一一对应。  
![[Pasted image 20240816093843.png]]  
receptive field一一对应的神经元参数都是一样的，每一个神经元对应的参数称为filter  

#### Result

![[Pasted image 20240816094246.png]]  

模型弹性减小，FC加上receptive field, 加上parameter sharing就是convolutional layer  

>到这里只讨论了卷积层

**使用了convolutional layer称为convolutional neural network，即CNN**  

### 另一种思路

在一个网络层里，设定一系列的filter，在图片中获取pattern    
![[Pasted image 20240816095040.png]]  

#### 如何获取pattern  

每个filter里的数值就是模型参数，通过训练固定了某种以数值表示的模式，当这种模式与RGB数值相乘时(图片信息)，获得了图片了pattern  
![[Pasted image 20240816095951.png]]  

- 如何相乘
	- 从左到右从上到下以固定stride遍历 （**称为convolution**）
	- 计算inner product（elementwise）
- 每个filter都遍历了整张图，得到一个**feature map**
	- ![[Pasted image 20240816100042.png|275]]
- 所有filter得到的feature map叠起来，看作是另一张图片，维度为filter的个数(feature map)
- 对这张新的图片进行下一轮卷积操作
	- 此时filter的超参数变成自己设定的大小和图片的channel数 (e.g. 3,3,64)
	- 而此时图片的channel数实际上是上一次卷积的filter数目，每次的filter数目是自己设定的
- 不断卷积直到获得足够的pattern信息
###### 关于kernel size
^3eb341
$(3\times 3)$大小的kernel size能否获取到足够的信息描述一个pattern  
可以这么理解  
在第一层卷积的时候，在原图大小是$(3\times 3)$  
在第二层卷积的时候，每个数值都包含着$(3\times 3)$大小的信息，所以再次进行$(3\times 3)$的卷积时，原图大小会变成$(5\times 5)$  
随着卷积层数，每次看到的原图信息会增加，不会因为kernel size过小而看不到足够信息  
但是应该会有损失

### Pooling

#### Observation 3

对图片进行subsampling不会影响图片的pattern（对应的object）  

> 把偶数列拿掉之类的

#### Simplication 3

pooling 更接近于activation function，对filter得到的feature map做subsampling  

例如max pooling  
![[Pasted image 20240816103826.png]]   
![[Pasted image 20240816103834.png]]  
- 选定一个范围为一组，选其中最大的

一般是几次convolution一次Pooling  
Pooling也会一定程度损害结果，所以可能也不用   

### Entire model  

![[Pasted image 20240816104423.png]]  

- flatten 把最后的结果拉平然后过一层FC
- 经过softmax计算概率得到分类结果

## Application

### Alpha GO

棋盘的现状作为输入，下一步落子位置作为输出  
当成分类问题的话就是多个棋盘位置的类别选一个类别下  

![[Pasted image 20240816104736.png]]  
- 棋盘用一个向量表示
- 使用FC可以做
- 但是CNN结果更好

#### CNN方法

- 把棋盘当成$19\times 19$的图片
- 原论文把每个位置使用48个channel表示(domain knowledge)

- CNN是图像特化的，使用需要与图片有相同的特性
	- 棋盘也有小的pattern  （原论文$5\times 5$作为第一个kernel size）
	- 相同的pattern可以出现在不同位置
- 但是棋盘不能subsampling
	- 所以直接不用pooling

![[Pasted image 20240816105605.png]]  

也可以用在语音，文字上  

## Problem 

普通的CNN没办法处理图像放大缩小或旋转的问题（data argumentation）  
这些变化改变了图片的数值，可能会影响结果  
- Filter 大小固定，学出来判断多大的pattern也是固定的
- 数据集里有可能可以
- 一些translation可能可以容忍（pooling消除掉了transform的影响）
- 但是总体上应该是 **invariant to scaling and rotation**

什么样的网络能够解决（现在应该有更多了）  
见[[Spatial Transformer Layer]]

