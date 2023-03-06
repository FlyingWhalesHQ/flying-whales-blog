---
layout: post
title:  "Perceptron"
date:   2023-03-02 18:14:54 +0700
categories: jekyll update
---

# TOC

- [Definition](#define)
- [Loss function](#loss)
- [Stochastic gradient descent](#sgrad)
- [Neural network](#nn)
- [Code example](#code)

# Definition <a name="define"></a>

Remember, from the linear function that takes linear combinations of parameters $$ \theta $$ and input x:

$$ \hat{y}=h_{\theta}(x) = \theta \dot x $$

Instead of the logistic function, we use a step function (with threshold) to wrap around that linear combination to have a new hypothesis function h:

$$ h_{\theta}(x)=step(\theta^{\top} x)= {\frac{0 if z < 0}{1 if z >= 0}} $$ 

Sometimes we use a sign function. A sign function is similar to a step function, it also use 0 as a threshold:

$$ sgn(z) = {\frac{\frac{-1 if z <0}{0 if z = 0}}{1 if z > 0 }} $$

Because of its step (and sign) function, this architecture is called a threshold logic unit. It is also the simplest artificial neural network architecture. Due to the nature of its output, this architecture can do classification, starting from a binary one.

# Loss function <a name="loss"></a>

Assume that the binary classification problem has a linear solution: the plane of inputs is linearly separable. One of the ways to go about this is to define a loss function that simply counts the number of misclassified points:

$$ L(\theta) = \sum_{x}(-y_{i} . sgn(\theta^{\top}x_{i})) = \sum_{x}(-y_{i} .sgn(\hat{y})) $$

X is the number of misclassified points. You can see that when x is misclassified, $$ y_{i} \neq \hat{y} $$ hence $$ -y_{i} . \hat{y} = 1 $$. When $$ L(\theta) = 0 $$ there is none misclassified. A downside of this simple loss function is that it is hard to imagine its derivative. Thinking in a similar vein, we try another loss function:

$$ L(\theta) = \sum_{x}(-y_{i} . \hat{y}) = \sum_{x}(-y_{i} \theta^{\top} x_{i}) $$

This is without the sign function. One thing about this multiplication loss function is that it heavily penalizes the misclassified points that are far from the decision line. In the previous loss function with the sign function, all misclassified points are penalized the same amount. 

# Stochastic gradient descent <a name="sgrad"></a>

We calculate the loss function for a single point of data, since it is how stochastic gradient descent is done:


$$ L(\theta;x_{i};y_{i}) = -y_{i} \theta^{\top} x_{i} $$ 


The derivative:


$$ \frac{\partial}{\partial \theta} L(\theta;x_{i};y_{i})= - y_{i} x_{i} $$


From there, let’s start the approximating process. Firstly, initialize a random parameter vector $$ \theta $$ . Choose one random instance (point) $$ x_{i} $$. If the point is classified correctly already, i.e. $$ sgn(\theta^{\top}x_{i}) = y_{i} $$, we don’t need to do anything. However, in this random scanning step, if the point $$ x_{i} $$ is misclassified, we will adjust the parameter vector $$ \theta $$ by the update rule:

$$ \theta \leftarrow \theta + \alpha y_{i} x_{i} $$

As we can see, with this update rule, we multiply the point with its label which means the algorithm would make a shift toward the correct parameter faster if the point is discovered to be farther from the decision line. After deciding the fate of that one point, we continue to search for misclassified points, if there is none, then we stop the algorithm. If we discover a misclassified point, come back to the part of adjusting the parameter vector  with the equation above.

The above update rule is for the sgn function. Following is the update rule for the step function. The same logic applies: when the prediction is incorrect, go and strengthen the connection (parameter) that could have made the correct prediction. In other famous words “Cells that fire together, are wired together”. This is an analogy to biology since these are similar to the biological neurons in the sense that when one neuron triggers another neuron often, the connection between them grows strong and they are considered to be in a group (or a net/network). Here is the update rule:

$$ \theta_{j} \leftarrow \theta_{j} + \alpha (y_{j} -\hat{y_{j}}) x_{i} $$

As you can see this is similar to linear regression, but the meaning is very different, since the domain of y in the classification problem is structurally different from the R-valued domain of y in linear regression. Perceptron’s domain is also different from the logistic regression output domain. 

# Neural network <a name='nn'></a>

Please have a look at the historical perceptron, inspired by biology, from the famous paper of Frank Rosenblatt (1958)

<img src='perceptron.png'>

The following picture shows the perceptron of today, as a threshold logic unit:

Source: Handson Machine Learning, O’Reilly

Usually, the step function is called an activation function, and for neural net, they use non linear activation functions, so that the entire network doesn’t end up being a big chunk of linear calculation. 

# Code example <a name="code"></a>
