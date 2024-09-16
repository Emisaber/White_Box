> `torchvision.transforms` 更新了，所以一部分代码可能得改成`torchvision.transforms.v2`  

When an image is transformed into a PyTorch tensor, the pixel values are scaled between 0.0 and 1.0.  

This transformation can be done using `torchvision.transforms.ToTensor()`. It converts the PIL image with a pixel range of $[0, 255]$ to a PyTorch FloatTensor of shape (C, H, W) with a range $[0.0, 1.0]$.   

After transform image to tensor, we may perform image normalization   

**Normalization**   

$$\bar x = \frac{x-\mu}{\sigma}$$    

- Normalizing the images means transforming the images into such values that the mean and standard deviation of the image become 0.0 and 1.0 respectively.   
- It helps get data within a range and reduces the skewness(偏斜), which helps learn faster and better.  
- It can help tackle the diminishing and exploding gradients problems

#### In Pytorch

`torchvision.transforms.Normalize()`  
Normalizes the tensor image with mean and standard deviation  

**Parameter**: 
- mean: Sequence of means for each channel
- std:  Sequence of std
- inplace: Whether operate data in-place

**Returns**:  Normalized Tensor Image  

To normalizing images in Pytorch, we need to   
1. Load and visualize image and plot pixel values uisng PIL
2. Transform image to Tensors using `torchvision.transorms.ToTensor()`
3. Calculate mean and standard deviation
4. Normalize the image
5. Visualize normalized image
6. verify normalization


Load the image  
```python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open(img_path)
img_np = np.array(img)

plt.hist(img_np.ravel(), bins=50, density=True)   # 拉伸为一维
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
```

It may look like this   
![Pasted image 20240916085717](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240916085717.png)  


Transforming images to Tensors  
```python
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# define custom transform function
transform = transforms.Compose([
	transforms.ToTensor
])

# transform PIL to Tensor
img_tr = transform(img)
img_np = np.array(img_tensor)

plt.hist(img_np.ravel()， bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
```

value transform into $[0., 1.]$  

![Pasted image 20240916090506](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240916090506.png)  

Calculate mean and std  
```python
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)
```

Calculated the mean and std of the image for all three channels Red, Green, and Blue.   

**For images that are similar with ImageNet, we can use mean and std instead  from ImageNet**    


**Normalizing the images**  
```python
from tochvision import transforms
# define custom transform function
transform_norm = transforms.Compose([
	trsanfroms.ToTensor(),
	transforms.Noeliiza(maean, std)
])  # mean和std是外面的变量

img_normalized = transform_norm(img)
 
img_np = np.array(img_normalized)

plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
```

![Pasted image 20240916093018](https://raw.githubusercontent.com/Emisaber/pic_obsidian/main/Pasted%20image%2020240916093018.png)  


Visualize the normalized image   
```python
img_normalized = transform_norm(img)

img_normalized = np.array(img_normalized)

img_normalized = img_normalized.transpose(1, 2, 0)

plt.imshow(img_normalized)
plt.xticks([])
plt.yticks([])
```


Calculate the mean and std again  
```python
img_nor = transform_norm(img)

mean, std = img_nor.mean([1,2]), img_nor.std([1,2])

print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)
```


### Reference

[How to normalize images in PyTorch ? - GeeksforGeeks](https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/)

