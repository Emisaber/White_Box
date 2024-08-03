### 待补
- [ ] Vanishing gradient

# Activation Function
For a given node, the inputs are **multiplied by the weights in a node and summed together**. This value is referred to as the **summed activation of the node**. The summed activation is then transformed via an activation function and defines the specific output or “**activation**” of the node.  
The complexity of activation function will increase the complexity of trainning  

## Typical activation function
#### Linear activation function
The simplest activation function is linear function   
$$\sigma  =  cx$$
it is easy to train but cannot learn complex mapping functions  

>Linear activation functions are still used in the output layer for networks that predict a quantity  

### Nonlinear

#### sigmoid
> sigmoid means s-shaped

For a long time, through the early 1990s, it was the default activation used on neural networks.  

sigmoid function, also called the **logistic function**.  
The input to the function is transformed into a value between **0.0 and 1.0**.  

details in [[sigmoid]]
#### hyperbolic tangent

outputs values between -1.0 and 1.0.   
In the later 1990s and through the 2000s, the tanh function was preferred over the sigmoid activation function as models that used it were easier to train and often had better predictive performance.  
better than sigmoid  

details in [[tanh]]  <-  miss
#### Limitation of sigmoid and Tanh

both sigmoid and tanh are **saturate**(饱和)，which means the large values snap to 1.0 and and small values snap to -1 or 0 for tanh and sigmoid respectively.  
the function only sensitive to changes around theri mid-point(0.5 for sigmoid and 0.0 for tanh) 

饱和情况下，只对中间小区间的数值变化敏感，哪怕输入数值的信息比较重要，输出的也是两端。这种情况下，梯度会同样变得极端(0)，神经元(激活函数)无法接受正确的梯度信息，难以通过反向传播来有效更新权重。  
ReLU 修正了这个问题 [How to Fix the Vanishing Gradients Problem Using the ReLU - MachineLearningMastery.com](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)  👈还没看   

#### ReLU
one of piecewise linear hidden units  (分段线性)
**Rectified(整流) linear activation unit  --- ReLU**  
networks that use the rectifier function for the hidden layers are referred to as **rectified networks**.

details in [[ReLU(Rectified Linear Unit)]]


## Reference
- [A Gentle Introduction to the Rectified Linear Unit (ReLU) - MachineLearningMastery.com](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)


