**pytorch**  

### Dataset & Dataloader
- Dataset
	- Dataset stores the samples and their corresponding labels
- Dataloader
	- wraps an iterable around the Dataset

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

pytorch offers domain-specific libraries such as `TorchText`, `TorchVision`, `TorchAudio` all of which include datasets.  
use `TorchVision` as example  

`datasets` module contains `Dataset` objects for real world datasets   
TorchVision `Dataset` includes two arguments: **`transform` and `target_transform` to modify the samples and labels respectively.**    

```python
traning_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
- use FashionMNIST dataset
- the code will download data from open datasets

Dataloader supports automatic batching, sampling, shuffling and multiprocess data loading.  
pass `Dataset` to `Dataloader`  
```python
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

### Creating models

For accelerating, move model to GPU or MPS is available
```python
# for accelerate, move model to GPU or MPS is available
device = (
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)
# print(f"using {device}")
```

To define a neural network in PyTorch, we create a class that inherits from `nn.Module`.  
```python
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nnFlatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
		)
	def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 

model = NeuralNetwork().to(device)
print(model)
```

