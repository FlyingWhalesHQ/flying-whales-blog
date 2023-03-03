---
layout: post
title:  "Linear Regression"
date:   2023-03-02 18:14:54 +0700
categories: jekyll update
---

# TOC

- [Introduction](#intro)
- [Gradient descent](#gradescent)
- [Maximum likelihood](#maxili)
- [Nonlinear linear](#nonlin)

# Introduction <a name="intro"></a>
This is the hello world of econometrics and machine learning. Before diving into the mathematical equations, let’s make some terms clear with an example. First, since there is a function, we would use it to map input into output. Output is the desired variable such as the price of a house that we would like to predict. Input would be the features of that house such as: age, floors, rooms, location, etc. If we have information about many houses, we can perform a linear regression on that dataset. The linear regression function would use one parameter per input to take into account the information that input carries so that the meaning of the input (age) carries over into the output. 

To understand what the name linear regression means: it is linear since the addition of the multiplication of parameters and inputs are linear operations. It is regression since it outputs a predicted value (the symbol y is with a hat). Here is the mathematical function:

$$  \hat{y} = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{n} x_{n}  $$

with n is the number of features, $$ x_{i} $$ is the $$ i_{th} $$ feature value, $$ \theta_{j} $$ is the $$ j_{ith} $$ model parameter ($$ \theta_{0} $$ is the bias term and $$ \theta_{1}, \theta_{2},...,\theta_{n} $$ are feature weights. There is a vector form for it:

$$ \hat{y}=h_{\theta}(x) = \theta \dot x $$

Where $$ h_{\theta} $$ is the hypothesis function with parameter vector .

Linear regression is among the parts of machine learning that reuses traditional econometrics (others include Bayesian inference, Maximum Likelihood). It is reused since the problem remains evergreen: given a dataset, how to find the fittest relationship that fits inputs x into the output y. To explain in better words: we care about some character y of a second hand car the most - in this case y is the price, therefore we collect and select data of its features (mileage, age, brand). With that dataset, we use mathematics to find the best possible function that maps from the input vector into its price. When we say best possible, we mean to reduce some sort of distance of the real ys and the computed function to the least. Still in traditional econometrics, the distance (also called error) is the Euclidean distance between the point y and the line. Then we take average and it is called mean square error (MSE), with $$ x^{(i)} $$ to be the vector of x at $$ i^{th} $$ instance, and X to denote the matrix containing all feature vectors of all instances in the dataset excluding labels y column vector:

$$ MSE (X, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (\theta^{\top} x^{(i)} -y^{(i)})^{2} = \frac{1}{m} ||y - X \theta ||_{2}^{2} $$

Since X and h are known, the problem becomes finding  to min MSE: $$ \theta^{*} = arg min_{\theta} MSE(\theta) $$. To minimize MSE, firstly, there is a closed form solution. A closed form solution is a solution derived by mathematical logic with mathematical symbols in it. An analytical solution is a solution by giving numerical values to prove its quality of solving the problem. Here is how to derive the closed form solution: we solve the optimization problem by letting its gradient be 0. For a linear model, this is straightforward: we compute the partial derivatives of the cost function with regard to the parameter j, then we put them all together in a big notation.
Partial gradient by j of the error function is calculated using chain rule:

$$ \frac{\partial}{\partial \theta_{j}}MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} 2(\theta^{\top} x^{(i)} - y^{(i)}). \frac{\partial}{\partial \theta_{j}}(\theta^{\top} x^{(i)} - y^{(i)}) $$

$$ = \frac{2}{m} \sum_{i=1}^{m}(\theta^{\top} x^{(i)} - y^{(i)}) x_{j}^{(i)} \forall j \in (1,n) $$

Sometimes they put a $$ \frac{1}{2} $$ as a multiplicator in MSE to cancel out the scalar 2, for convenient. The gradient by $$ \theta $$ of the error function in the vector notation would be:

$$ \nabla_{\theta}MSE(\theta){\partial \theta} = \frac{2}{m} X^{\top} (X \theta - y) = 0 $$

$$ \Leftrightarrow X^{\top} X \theta = X^{\top} y $$

$$ \Leftrightarrow = (X^{\top} X)^{-1} X^{\top} y $$

(this is known as the normal equation)


Apart from MSE, we use some other function that fits our modern purpose in some other way. And we call that class of functions cost function (or loss function). To show an example, another useful loss function is MAE (mean absolute error):

$$ MAE (X, h) = \frac{1}{m} \sum_{i=1}^{m} | h(x^{(i)}) - y^{(i)} | $$

We can see that, instead of taking the Euclidean norm, it just minus the real y from the predicted value y and takes the positive value of that difference. With the square and square root way of calculation, the RMSE is more sensitive to outliers.

In the next section, we look at the second approach to find the minimum: using gradient descent as a search algorithm.


# Gradient descent <a name="gradescent"></a>

In econometrics, if we can approximate the distribution of the population by estimating the parameters of the function of the sample data then we are done. Since the sample is the representative part of the population. Voila, we found and are now able to describe the interested population well. For example: we have the population of smokers in a given region of the world, then we can suggest policy in the direction of improving people’s health in that region. In machine learning, however, the purpose is different since we are somewhat closer to the market: we need to generate predictions and those predictions need to be good enough. In other words, the algorithm needs to work well on unseen data (unforeseen future). We also live in an era where the dataset has become big - so big at the scale that researchers haven’t seen before. Together with the technological advance that gives us computing resources a lot more freedom than in decades ago - at the beginning of machine learning conception. With that in mind, we can continue to explore machine learning as a powerful and pragmatic tool in the modern world, at the same time keeping mathematics and other closely related fields in our toolbelt. 

One of the highlights of this modern field is that we use an optimization approach called gradient descent to iteratively change parameters to minimize the loss function. And we use Python. Imagine a general function of any shape (with mountains and valleys). The gradient descent is a technique that guides along the slope of the graph (assumed loss function) until it reaches a minimum.

<img src='https://blog.paperspace.com/content/images/2018/05/challenges-1.png'>

<p style="font-size:9px">Source: https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/</p>

Assume that the above terrain is a loss function’s graph, when we see it and ask the question how to achieve that minimum, we can come up with the following procedure:
<ul>

<li>Randomly initialize the parameters (weights$$\theta$$) </li>
<li>Since we need to move in the direction of the steepest possible descent (hence the name), we update the parameter vector with: $$ \theta_{j} \leftarrow \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} Loss(\theta), \forall j \in (1,n) $$ (we descend by one parameter/dimension at a time). This is equivalent to: $$ \theta \leftarrow \theta - \alpha \frac{\partial}{\partial \theta}Loss(\theta) $$</li>
<li>Stop when converged</li>
</ul>

Full gradient descent takes into account the full training dataset, but it is computationally costly. Another way to do it is to take into account only a mini batch of the training set each time. Stochastic gradient descent is when the algorithm takes into account only one instance of the training data at a time.

Notice that $$ \alpha $$ can be tuned to be constant (usually at 0.1) or it can be adaptive ($$ \alpha = \frac{1}{updates so far} $$).

# Maximum likelihood <a name="maxili"></a>

This is also a classical technique to demonstrate that the loss function of least squares is a natural and obvious implication of the linear regression model.

Back to the linear equation, this time with an error term (this is econometrics!)

$$ y^{(i)} = \theta^{\top} x^{(i)}+ \epsilon^{(i)} $$

$$ \epsilon $$ contains both unknown inputs and random noise. For this technique, we need to assume $$ \epsilon^{(i)} $$ are iid (independently and identically distributed) according to a Gaussian distribution with mean 0 and variance $$ \sigma^{2} $$.

$$ p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(\epsilon^{(i)})^{2}}{2 \sigma^{2}}) $$

$$ \Rightarrow p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^{\top} x^{(i)})^{2}}{2\sigma^{2}}) $$

We need to find a way to maximize the probability of y that was given in relation with x and $$ \theta $$. This way involves finding parameters, hence we can rewrite:

$$ L(\theta) = L(\theta;X,y) = p(y|X;\theta) $$

This is to find $$ \theta $$ that maximize the joint distribution of X (so that y happens the most):

$$ L(\theta) = \prod_{i=1}^{n} p(y^{(i)}|x^{(i)};\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(y^{(i)} - \theta^{\top} x^{(i)})^{2}}{2\sigma^{2}}) $$

Maximizing $$ L(\theta) $$ is similar to maximizing the log likelihood:

$$ l(\theta) = log L(\theta) = log \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(y^{(i)} - \theta^{\top} x^{(i)})^{2}}{2\sigma^{2}}) $$

$$ = \sum_{i=1}^{n} log \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(y^{(i)} - \theta^{\top} x^{(i)})^{2}}{2\sigma^{2}}) $$

$$ = n log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^{2}} \frac{1}{2} \sum_{i=1}^{n}(y^{(i)} - \theta^{\top} x^{(i)})^{2} $$

Taking out all the scalar values in the above function, we have MSE. It means that when we find the maximum likelihood of $$ \theta $$ for maximizing the joint probability of X, we come to the least square error function. That’s it for econometrics!


# Non-linear linear <a name="nonlin"></a>

When we add a quadratic term of x, if the parameters $$ \theta $$ still come together as a linear combination, the problem can still be solved as a linear regression in general.

Code: …


Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
