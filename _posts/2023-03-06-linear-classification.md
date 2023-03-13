---
layout: post
title:  "Linear Classification"
date:   2023-03-06 2:30:54 +0700
categories: jekyll update
---

# TOC
- [Introduction](#intro)
- [Code example](#code)

# Introduction <a name="intro"></a>

Remember the linear combination of weights $$\theta$$ and input features x:

$$ \hat{y}=h_{\theta}(x) = \theta \cdot x $$

Think about all the times we wrap this linear combinations in a nonlinear function to activate it (it here means the linear combination, it could be translated as to activate a neuron in deep learning context). In this post we would browse different nonlinear functions that ouput classes instead of a value, to serve the problem of classifying input data into classes. Let's consider binary classifers.

## Sign function

A sign function is named so since it outputs the sign of the value of the linear combination:

$$ sign(\hat{y}) = 
\begin{cases}
        -1 & \text{if $\hat{y}$ < 0}\\
        +1 & \text{if $\hat{y}$ > 0}\\
        0 & \text{if $\hat{y}$ = 0}\\
\end{cases}
$$

The decision boundary according to x for this classification problem is $$ \hat{y} = \theta \cdot x = 0 $$

## Loss functions and their gradients

### Zero-one loss function

The loss function for the sign function is a zero-one loss function. Which means that if we output the correct sign, the loss is 0, if we output the incorrect sign, the loss is 1. The loss function for a single point of the data is: 

$$ Loss_{0-1}(x,y,\theta) = 1[\hat{y} \ne y] $$

$$ 1[...] $$ is a function such that if the inside statement is true, it returns value 1, if false, it returns 0. For the whole training dataset, we simply take average of those smaller loss functions. There are two more concepts that we need to know: the score and the margin. The score is $$ \theta \cdot x $$ and it is how confident we are in predicting the positive class. The margin is $$ (\theta \cdot x)y = \hat{y} y $$ and it shows how correct we are. Since the margin takes into account the largeness of prediction and true value, we use it as the loss function instead of the not-equal comparison. Actually they are equivalent:

$$ Loss_{0-1}(x,y,\theta) = 1[\hat{y} \ne y] $$

$$ \Leftrightarrow Loss_{0-1}(x,y,\theta) = 1 [(\theta \cdot x) y <= 0 ] $$

<img width="339" alt="LinClass" src="https://user-images.githubusercontent.com/7457301/224597096-37846d67-d9a6-4940-bffb-d6325cb3657c.png">
<p style="font-size:9px">Source: https://stanford-cs221.github.io/autumn2022/modules/module.html#include=machine-learning/linear-classification.js&mode=print1pp</p>

Now we can plot the loss function, for all margin values that are negative, the loss is 1. For all margin values that are non negative, the loss is 0. Remember that the total loss of the whole training set is the average of all the loss zero-one function for each data point, hence the gradient of the total loss depends on the total gradient of all smaller loss functions, apart from the scalar of dividing by the size of the training set (for averaging calculation). Here is the total of smaller gradients:

$$ \nabla_{\theta} Loss_{0-1}(x,y,\theta) = \nabla 1 {[(\theta \cdot x) y <= 0 ]} $$

Gradient of this loss function is almost zero everywhere, except for the point where margin = 0, at that point gradient is infinite. This is hard in updating the parameters hence optimization.

### Hinge loss function

To account for that, we use the hinge loss function:

$$ Loss_{hinge} (x,y,\theta) = max\{1 - margin, 0\} $$

When margin >= 1 (when we are correct), the loss function equals 0. When margin < 1, the loss decreases gradually (linearly). Gradient of the hinge loss is better than the zero-one loss function:

$$ \nabla Loss_{hinge}(x,y,\theta) = 
\begin{cases}
    -xy & \text{if 1 - margin > 0}\\
    0 & \text{otherwise}
\end{cases}
$$

We said 0 otherwise since it is mostly otherwise. At the exact 0, the gradient doesn't exist since the left derivative and the right derivative don't match (the function is abrupt at that point).

Let's compare zero-one loss and hinge loss. In zero-one loss, the prediction is exact: either you hit or you miss, and you score accordingly. But in hinge loss, we don't just use the margin, we use $$ 1 - margin $$, so we aim for some positive margin even when we already predict correctly. That changes the loss a bit fundamentally.

### Logistic loss

Logistic loss function graph descends smoother than hinge loss function graph. And this loss functions only goes towards 0, it never reaches 0.

$$ Loss_{logistic}(x,y,\theta) = log(1+e^{-(\theta \cdot x)y}) $$

# Code example <a name="code"></a>

|$$x_1$$|$$x_2$$|y|
|--|--|--|
|3|-1|+1|
|2|3|+1|
|-1|0|-1|

|$$\theta_1$$|$$\theta_2$$|
|--|--|
|0.5|3|



```python
import numpy as np
x=[[3,-1],[2,3],[-1,0]]
theta=[[.5],[3]]
yhat=np.dot(x,theta)
yhat
```




    array([[-1.5],
           [10. ],
           [-0.5]])



|$$ x_1 $$|$$ x_2 $$|y|$$ h_{\theta} $$|$$ \hat{y}=sgn(h_{\theta}) $$|zero-one loss|
|--|--|--|--|--|--|
|3|-1| -1.5|+1|-1|1|
|2|3| 10|+1|+1|0|
|-1|0| -0.5|-1|-1|0|

training loss = $ \frac{1}{3}$

|$$ x_1 $$|$$ x_2 $$|y|$$ h_{\theta} $$|$$ \hat{y}=sgn(h_{\theta}) $$|zero-one loss|margin=$ \hat{y}{y} $|hinge|
|--|--|--|--|--|--|--|--|
|3|-1| -1.5|+1|-1|1|1.5|0|
|2|3| 10|+1|+1|0|10|0|
|-1|0| -0.5|-1|-1|0|0.5|0.5|

training loss = $ \frac{1}{6}$
