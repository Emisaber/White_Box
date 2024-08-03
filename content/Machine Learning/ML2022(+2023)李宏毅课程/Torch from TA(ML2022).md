训练，验证，测试
## Dataset & DataLoader
`torch.Dataset` 存储数据集  
`torch.DataLoader` 加载数据集，将数据集打包成batch，同时底层支持平行化计算
```python
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
	def __init__(self, file):  # 初始化数据集或进行预处理后读取
		self.data = ...
		
	def __getitem(self, index): # 索引
		return self.data[index]
		
	def __len__(self):          # 返回数据集大小
		return len(self.data)

dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
```
- DataLoader 通过连续调用`__getitem__` 来得到一个batch
### Tensors
high-dimensional matrices(arrays)    
`.shape` for dimension  
pytorch的dim 与Numpy 的axis 基本一样  
#### Creating Tensors
- from python list or numpy array(ndarray)
```python
x = torch.tensor([],[])
x = torch.from_numpy(np.array([[], []]))
```

- constant
```python
x = torch.zeros([]) #shape
x = torch.ones([]) #shape
```

#### calculation
- addition
- subtraction
- power `x.pow()`
- sum `x.sum()`
- mean `x.mean()`
- 转置 transpose `x = x.transpose(0,1) # 将第0和第1个维度置换`
- squeeze  `x.squeeze(dim)` 对某一个维度进行压缩
	- 将维度的值为1的部分去掉
	- `torch.Size([1, 2, 3])` -> `x.squeeze(0)`  ->`torch.Size([2, 3])`
- unsqueeze `x.unsqueeze(dim)` 
	- 扩展在这个维度扩展一个数值为1的维度
	- `torch.Size([2, 3])` -> `x.squeeze(0)` ->`torch.Size([1, 2, 3])`
- `torch.cat` `concatence = torch.cat([x, y, z], dim=1)`
	- 在指定维度的(在这里第一维)方向上连接
	- `x torch.Size([2, 1, 3])` `y torch.Size([2, 3, 3])` `z torch.Size([2, 2, 3])`  则 concatence `torch.Size([2, 6, 3])`

#### Data Type
- int 
- float
- 详见官方文档
#### 与numpy
- `.shape`
- `.dtype()`
- `.reshape()`
- `.squeeze()`
- `.unsqueeze()` in torch and `np.expand_dims(x, 1)` in numpy

#### Device
`tensor.to()`
`.to('cpu')` 或 `.to('cuda')`

检查是否能用GPU  
```python
torch.cuda.is_available()
multiple GPUs specify 'cuda:0', 'cuda:1', 'cuda:2'
```

#### Gradient calculation
```python
x = torch.tensor([[1., 0], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
x.grad
```


## Trainning
`torch.nn`  
linear layer 为例  
![[Pasted image 20240730144256.png]]
```python
layer = torch.nn.Linear(32, 64)  # 输入是32维，输出是64维
layer.weight.shape  (64, 32)
```

`nn.sigmoid()`， ``nn.ReLu()

#### Build neural network
```python
import torch.nn as nn
class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(10, 32)
			nn.Sigmoid(),
			nn.Linear(32, 1)
		)
	def forward(self, x):
		return self.net(x)
```
- 定义了一个三层的模型，第一层是线性层，第二层是sigmoid，第三层是线性层
- 需要定义一个`forward()`，定义这个网络怎么计算
- 这里是用sequential套起来，有其他的方法，这个得看文档或其他人怎么写

其他方法  
```python
class MyModel(nn.Module):
	def __init__(Self):
		super(MyModel, self).__init__()
		self.layer1 = nn.Linear(10, 32)
		self.layer2 = nn.Sigmoid(),
		self.layer3 = nn.Linear(32, 1)
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		return out
```
- 这里就纯手写
- 等效于上面

#### Loss Function
MSE(mean Squared Error)  -- regression tasks
```python
criterion = nn.MSELoss()
```

Cross Entropy  -- classification tasks
```python
criterion = nn.CrossEntropyLoss()
```

计算loss  
```python
loss = criterion(model_output, expected_value)
```

#### Optimization
`torch.optim`  

最基本的梯度下降  SGD  
```python
optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)
```

定义好优化器后  
1. call `optimizer.zero_grad()` 重置模型参数的梯度
2. call `loss.backward()` 反向传播计算梯度
3. call  `optimizer.step()` 调整参数

## Entire Procedure
#### 训练  
```python
# 定义数据集类...
# 定义模型.....
# 然后
dataset = MyDataset(file)  # 取数据
train_set = DataLoader(dataset, 16, shuffle=True) # 加载数据
model = MyModel().to(device) # 初始化模型并转移到GPU
criterion = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.1)  #定义optimizer


for epoch in range(n_epochs):  # 遍历所有epoch
	model.train()   # 设置为训练模式
	for x, y in train_set: 
		optimizer.zero_grad()
		x, y = x.to(device), y.to(device)  # 转移到GPU才能算
		pred = model(x)
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()  
```
- 一个batch更新一次，一个epoch更新多次
#### 验证
```python
model.eval()
total_loss = 0
for x, y in eval_set:  # 数据集用相同的方式得到 
	x, y = x.to(device), y.to(device)
	with torch.no_grad():  # 取消梯度计算
		pred = model(x)
		loss = criterion(pred, y)
	total_loss += loss.cpu().item()*len(x)  # 转移到cpu方便处理
	avg_loss = total_loss / len(eval_set.dataset)
```

#### 测试
```python
model.eval()
preds = []
for x in test_set:
	x = x.to(device)
	with torch.no_grad():
		pred = model(x)
		preds.append(pred.cpu())
```

#### 补充
- `model.eval()` 与 `model.train()`
	- 有一些模型这两步有很大区别，需要区分
- `with torch.no_grad()` 取消梯度避免因为程序问题导致多余的训练
	- 还会更快

#### 保存
Save
```python
torch.save(model.state_dict(), path)
```

Load
```python
ckpt = torch.load(path)
model.load_state_dict(ckpt)
```

## Colab
使用命令行时  
使用 `!` 会生成一个新shell，执行完kill掉  
使用 `%` 会保持对整个笔记本的影响，称 `magic command`  
例如当`cd`的时候应该用 `%`  
more magic command: [Built-in magic commands — IPython 8.26.0 documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html)  

在colab下载文件  
```colab
!gdown --id '链接内的id' --output 重命名
```
命令行操作创建文件夹之后再下  

