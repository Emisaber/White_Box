$$sigmoid(x) = \frac{1}{1+e^{-x}}$$
or   
$$sigmoid(x) = \frac{e^x}{e^x + 1}$$
- s-shaped -- 拟合折线和曲线
- logistic function 逻辑函数  
- 可以用于分类  -- 计算概率

![[Pasted image 20240622110029.png|375]]
- 当z无穷大时 为1
- 当z无穷小时 为0
- 中间增长的速度(斜率)由z的系数决定
- 形状像s的**光滑**曲线
- 输出在01之间，用作概率

传入一元线性函数  
$$c\ sigmoid(b+wx) = c \frac{1}{1+e^{-(b+wx)}}$$
通过改变z的系数就能改变斜率，增加常数就能改变位置，中间实际上就是e的指数的形状，也就是一条直线  
所以将需要拟合的直线段(折线的一部分或者曲线的一部分)放进来，就能表示出那一段折线/曲线  
- **改变w，改变斜率**
- **改变b，改变截距(直线的截距，所以反映出来的是sigmoid左右移动)**
- **改变c，改变最值，0保持，最大值改变**

多个sigmoid表示piece curve  
$$y = b+\sum_ic_i\ sigmoid(b_i+w_ix_i)$$

传入多元线性   
$$y = b + \sum_ic_i\ sigmoid(b_i+\sum_jw_{ij}x_j)$$
简写(向量化)  
$$\textbf{r} = \textbf{b} + \textbf{w}\textbf{x}$$
$$y = b + \textbf{c}^Tsigmoid(\textbf{b} + \textbf{w}\textbf{x})$$  

## references
- [Site Unreachable](https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/)
- [Site Unreachable](https://www.analyticsvidhya.com/blog/2023/01/why-is-sigmoid-function-important-in-artificial-neural-networks/#:~:text=The%20sigmoid%20is%20a%20mathematical%20function%20t%20hat,useful%20for%20binary%20classification%20and%20logistic%20regression%20problems.)

