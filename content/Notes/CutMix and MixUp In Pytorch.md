### MixUp and CutMix
Mixup： blend images together   
CutMix:  Cut off a part of pixel and fill with pixel from other images 

Mixup 完全结合图像的信息，引入不自然的信息可能性比较大   
CutMix 结合图像的部分信息，能够加快训练效率  
![Pasted image 20240916103150](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240916103150.png)  


### In Pytorch

```python
import torch
from trochvision.transforms import v2
```

define number of classes  
```python
NUM_CLASS = 100
```

**A typical image classification pipeline**  
```python
preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
])

#dataset = FakeData(size=1000, num_classes=NUM_CLASSES, transform=preproc)

#img, label = dataset[0]
#print(f"{type(img) = }, {img.dtype = }, {img.shape = }, {label = }")
```

#### Add Mixup and CutMix  

**After DataLoader**  
DataLoader has already batched the images and labels for us, and this is exactly what these transforms expect as input  
```python
dataloader = DataLoader(dataset, batch_size, shuffle=true)

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

for images, labels in dataloader:
	images, labels = cutmix_or_mixup(images, labels)
```

The shape of tensor  
```
Before CutMix/MixUp: images.shape = torch.Size([4, 3, 224, 224]), labels.shape = torch.Size([4])
After CutMix/MixUp: images.shape = torch.Size([4, 3, 224, 224]), labels.shape = torch.Size([4, 100])
```

Label transform from `(batch_size)` into `(batch_size, num_class)`   
The transformed labels can still be passed as-is to a loss function like `torch.nn.functional.cross_entropy()`.   直接能按原样传入   

> for `cross_entropy()`, the target label shape can be (N, C), where C is the number of classes, and N is the batch size


**As part of the collation function**  

Add cutmix_or_mixup directly after the DataLoader is the simplest way, but it does not take advantage of the DataLoader multi-processing.  For that, we can pass those transforms as part of the collation function.   

```python
from torch.utils.data import default_collate
def collate_fn(batch):
	return cutmix_or_mixup(#default_collate(batch))

dataloader = DataLoader(datasetm batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

for images, labels in dataloader:
	pass
```

```
images.shape = torch.Size([4, 3, 224, 224]), labels.shape = torch.Size([4, 100])
```


**With non-standard input format**  
typical format is `(images, labels)`   
MixUp and CutMix will magically work by default with most common sample structures: tuples where the second parameter is a tensor label, or dict with a `label[s]` key.  

if samples have a different structure, use CutMix and MixUp by passing a callable to the labels_getter parameter.   

```python
def labels_getter(batch):
    return batch["target"]["classes"]

out = v2.CutMix(num_classes=NUM_CLASSES, labels_getter=labels_getter)(batch)
```

### Reference
- [How to use CutMix and MixUp — Torchvision main documentation](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html)


