---
layout: post
title:  "Probability"
date:   2023-04-20 10:14:54 +0700
categories: Mathematics
---

# TOC

- [Introduction](#intro)
- [Random Continuous Variable](#random)
- [Random Discrete Variable](#discrete)
- [Joint Probability](#joint)
- [Conditional Probability](#cond)
- [Expectation](#expect)


# Introduction

Probability is a branch of mathematics that deals with the study of random events and the likelihood of their occurrence. It is used to model situations where there is uncertainty or randomness involved, and is widely applied in various fields such as statistics, finance, physics, engineering, and computer science. Probability is also widely used in machine learning and artificial intelligence, where it is used to model uncertainty in data and to make predictions.

# Random variable

A randome variable x denotes an uncertain quantity. It may be the result of a coin flip or the measurement of temperature. Each time we experience x, it can take a different value $$ x_i $$. However, values can repeat themselves and some seems to appear more frequent than others. This information is captured by the probability distribution $$ Pr(x) $$ of the random variable x. Note that a random variable x can assign number to each outcome. For example, in the die experiment we can assign to the six outcomes i the numbers 10i: $$ x(1) = 10, x(2) = 20.. x(6) = 60 $$ or we can assign number 1 to even outcomes and number zero to odd outcomes: $$ x(1) = x(3) = x(5) = 0, x(2) = x(4) = x(6) = 1 $$.

We can also say, in some other words: if the experiment is done n times and the event A occurs $$ n_A $$ times, then with a high degree of certainty, the relative frequency $$ \frac{n_A}{n} $$ of the occurrence of A is close to P(A): $$ P(A) \approx \frac{n_A}{n} $$ provided that n is sufficiently large. In the limit, theoretically, the probability P(A) of event A can be described as a hypothesis $$ P(A) = lim_{n\to \infty} \frac{n_A}{n} $$.

The random variable can become a function f(x) when the domain is the set of all experiment outcomes (so that the total proabilities summed to one). Note that a function is a rule of correspondence between x and y, with the values of the independent variable x form a set D named the domain and the values of dependent y = f(x) form the range set R of the function. In another way, we have two sets of number D and R. For every x in D we assign a number y = f(x) belong to R. We would say f is the function of x. The mapping between x and y can be one to one or many to one.

There are two types of random variables: discrete and continuous. A discrete variable has a set of values. This set can be an ordered set, for example the list of a dice rolling values, ranging from 1 to 6, or it can be an unordered one, say, the weather outcomes of sunny, snowy, rainy and windy. It can be finite or infinite and the probability distribution is best shown as a histogram. With that, each possible outcome has a positive probability and the sum of all such probability is 1. On the other hand, continuous random variable has values in the real domain. These can also be finite or infinite, depending on the problem. It can be infinite but bounded and the probability distribution is best shown as the graph of the probability density function (pdf). Each outcome would have its own probability (propensity) and the integral of the pdf always be 1, similar to the discrete variable.

<img width="381" alt="Screen Shot 2023-04-22 at 17 09 39" src="https://user-images.githubusercontent.com/7457301/233777739-a4e04122-8b32-4dab-84c1-bef34260bad2.png">
<img width="411" alt="Screen Shot 2023-04-22 at 17 09 44" src="https://user-images.githubusercontent.com/7457301/233777742-9c6e74a2-0193-4e5c-9c80-26d1433f9d16.png">

Image: the visualization of the probability distribution of discrete and continuous variable

# Continuous random variable

## Normal (Gaussian) distribution
This is the most popular distribution. We say x is a normal (or Gaussian) random variable with parameters $$ \mu $$ and $$ \sigma^2 $$ if the density function is:

$$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2 / 2\sigma^2} $$

Many natural phenomena follows Gaussian distribution. One example, Maxwell arrived at the normal distribution for the distribution of velocities of molecules, under the assumption that the probability density of molecules with given velocity components is a function of their velocity magnitude and not their directions. 

## Exponential distribution

We say x is exponential with parameter $$ \lambda $$ if the density function is 

$$ f(x) = 
\begin{cases}
    \lambda e^{-\lambda x}, x \geq 0 \\
    0 \text{ otherwise} \\
\end{cases}
$$

Some exponentially distributed events are phone calls or bus arrivals, given that the occurrences of those events are independent.

<img width="363" alt="Screen Shot 2023-04-23 at 14 23 22" src="https://user-images.githubusercontent.com/7457301/233825907-9be4b608-eb3a-4be2-9134-3e5aa868f5aa.png">

Image: The waiting time at bus stop or phone calls, according to exponential distribution assumption

## Gamma distribution
We say x to be a gamma random variable with parameters $$ \alpha > 0, \beta > 0 $$ if

$$ f(x) = 
\begin{cases}
    \frac{x^{\alpha - 1}}{\Gamma(\alpha) \beta^{\alpha}} e^{-x/\beta}, x \geq 0 \\
    0 \text{ otherwise} \
\end{cases}
$$ 

with $$ \Gamma(\alpha) = \int_{0}^{\infty} x^{\alpha - 1} e^{-x} dx $$

The gamma distribution (which was mentioned) takes on different shapes and sizes.

## Chi-square distribution

x is said to be a $$ \chi^2(x) $$ with n degrees of freedom if 

$$ f(x) = 
\begin{cases}
    \frac{x^{n/2 - 1}}{2^{n/2} \Gamma(n/2)}e^{-x/2}, x \geq 0 \\
    0 \text{ otherwise} \\
\end{cases}
$$

with n = 2, we have the exponential distribution. 

## Uniform distribution

x is said to be uniformly distributed in the interval (a,b) $$ - \infty < a < b < \infty $$ if

$$ f(x) = 
\begin{cases}
    \frac{1}{b-a}, a \leq x \leq b \\
    0 \text{ otherwise} \\
\end{cases}
$$


<img width="340" alt="Screen Shot 2023-04-23 at 14 34 03" src="https://user-images.githubusercontent.com/7457301/233826351-8f8889bf-7208-4247-a7bf-ffdee967e096.png">

Image: A uniform distribution

## Beta distribution

The random variable x has beta distribution with nonnegative parameters $$ \alpha $$ and $$ \beta $$ if

$$ f(x) =
\begin{cases}
    \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1-x)^{\beta - 1}, 0 < x < b \\
    0 \text{ otherwise} \\
\end{cases}
$$

where the beta function $$ B(\alpha, \beta) = \int_0^1 x^{\alpha - 1} (1 - x)^{\beta - 1} dx = 2 \int_0^{2x} (sin \theta)^{2\alpha - 1} (cos \theta)^{2\beta - 1} d\theta $$

## Cauchy distribution

$$ f(x) = \frac{\alpha/\pi}{(x-\mu)^2 + \alpha^2}, \mid x \mid < \infty $$

## Laplace distribution

$$ f(x) = \frac{\alpha}{2} e^{-\alpha \mid x\mid}, \mid x \mid < \infty $$

## Maxwell distribution

$$ f(x) =
\begin{cases} 
    \frac{4}{\alpha^3 \sqrt{\pi}}x^2 e^{-x^2 / \alpha^2}, x \geq 0 \\
    0 \text{ otherwise} \\
\end{cases}
$$

# Discrete variable

## Bernouli distribution

The Bernoulli distribution refers to any experiment with only two possible outcomes: success or failure (head or tail). x is said to be Bernoulli distributed if x takes the values 1 and 0 with P(x=1) = p and P(x=0) = q = 1 - p

## Binomial distribution

When we have independent trial of n Bernoulli experiment, we call it a binomial random variable. x is said to be a binomial random variable with parameters n and p if x takes the values of n classes: 1, 2,..n with $$ P(x=k) = \binom nk p^k q^{n-k} $$ with p+q=1 and k = 1,2,..n

Since the binomial coefficient $$ \binom nk = \frac{n!}{(n-k)!k!} $$ grows rapidlly with n, it is difficult to compute the probability. So we can approximate this distribution with normal approximation and Possion approximation.

Let $$ n\to \infty $$ with fixed p. Then for k in the $$ \sqrt{npq} $$ neighborhood of np, we approximate $$ \binom nk p^kq^{n-k} \approx \frac{1}{\sqrt{2\pi npq}} e^{-(k-np)^2/2npq} $$ with p+q=1.

Let's state the law of large numbers: if an event A with P(A) = p occurs k times in n trials, then $$ k \approx np $$. In fact, $$ P(k=np) \approx \frac{1}{\sqrt{1\pi npq}} \to 0 $$ as $$ n \to \infty $$.

For the Poisson approximation: if $$ n \to \infty, p \to 0 $$ such that $$ np \to \lambda $$, $$ \frac{n!}{k!(n-k)!} p^k q^{n-k} \to e^{-\lambda} \frac{\lambda^k}{k!} $$ with k = 0,1,2,...

## Poisson distribution
The Poisson distribution represents random variables such as number of telephone calls for a fixed period, the number of winning ticketss in a large lottery, and the number of printing errors in a book. The event can be rare, but does happen. x follows a Poisson distribution with parameter $$ \lambda $$ if x takes the values 0, 1, 2, ... $$ \infty $$ with $$ P(x=k) = e^{-\lambda} \frac{lambda^k}{k!} $$, k = 0,1,2..$$\infty$$.

## Geometric distribution
Let x be the number of trials needed to find the first success in repeated Bernoulli trials. Then x follows a geometric distribution.

$$ P(x=k) = pq^{k-1} $$ with k = 1,2,3,..$$\infty$$.

The probability of event (x>m) is: $$ P(x>m)= \sum_{k=m+1}^{\infty} P(x=k)=\sum_{k=m+1}^{\infty} pq^{k-1} = pq^m(1+q+...) = \frac{pq^m}{1-q}=q^m $$
 
## Negative binomial distribution
x follows negative binomial distribution with parameters r and p if $$ P(x =k) = \binom {k-1}{n-1} p'q^{k-1} $$

## Discrete uniform distribution
P(x=k) = \frac{1}{N} with k = 1,2,..N

# Joint probability

Joint probability of variable x and y $$ Pr(x,y) $$ is the probability at which those two appear together. The summing of all outcome probabilities is still one as usual. When we concern multiple variables, we write $$ Pr(x,y,z) $$ for the joint probability of x, y and z. Or we write $$ Pr(\textbf{x}) $$ for the joint probability of all of the elements of the multidimensional variable $$ \textbf{x} = [x_1, x_2..x_K] $$. Similar for $$ Pr(\textbf{x}, \textbf{y}) $$.

To extract the probability distribution of a single variable from a joint distribution we sum (or integrate) over all other variables:

$$ Pr(x) = \int Pr(x,y) dy $$ for continuous y.

$$ Pr(x) = \sum_y Pr(x,y) $$ for discrete y.

Pr(x) is called the marginal distribution and doing the equation is called the marginalization process. 

<img width="223" alt="Screen Shot 2023-04-22 at 17 47 09" src="https://user-images.githubusercontent.com/7457301/233779591-9a121d26-b4fd-4b36-840f-39d5a319611b.png">

Image: Joint probability of two continuous variables x and y


# Conditional proability

The conditional probability is the probability of x condition on $$ y = y^* $$. This sentence is written mathematically as $$ Pr(x \mid y = y^*) $$. The thing is, the various probabilities of x given a specific y doesn't sum up to 1. So we normalize by the sum of all the probabilities in the slice so that the conditional probabilities become a distribution:

$$ Pr(x\mid y=y^*) = \frac{Pr(x,y=y^*)}{\int Pr(x,y=y^*) dx} = \frac{Pr(x,y=y^*)}{Pr(y=y^*)} $$

The denominator is the marginal probability of $$ y= y^* $$. The above is also equivalent to:

$$ Pr(x\mid y) = \frac{Pr(x,y)}{Pr(y)} $$

<img width="548" alt="Screen Shot 2023-04-22 at 17 47 16" src="https://user-images.githubusercontent.com/7457301/233779600-94cf75ce-f7a4-4e03-92bf-b3e058703ec3.png">

Image: Conditional probability of variable x given two values of y

# Bayes' rule

Since $$ Pr(x,y) = Pr(y\mid x)Pr(x) $$, we also have $$ Pr(x,y) = Pr(y\mid x)Pr(x) $$. Combining them we have $$ Pr(y\mid x) Pr(x) = Pr(x\mid y) Pr(y) $$.

$$ Pr(y\mid x) = \frac{Pr(x\mid y)Pr(y)}{Pr(x)} = \frac{Pr(x\mid y) Pr(y)}{\int Pr(x,y) dy} $$.

This is called the Bayes' rule and $$ Pr(y\mid x) $$ is called the posterior - what we know about y after taking x into account. The Pr(y) is the prior - what we know about y before considering x. $$ Pr(x\mid y) $$ is called the likelihood. Pr(x) is the evidence. So the posterior is equal to the likelihood multiplied by the prior adjusted for the evidence.

# Independence

Independence is a condition that knowing x doesn't give out information about y. Hence the conditional probability is simply the evidence $$ Pr(x\mid y) = Pr(x) $$. The joint probability then becomes the product of the marginal probabilities $$ Pr(x,y) = Pr(x\mid y) Pr(y) = Pr(x) Pr(y) $$. Given two independent and mutually exclusive events A and B, then $$ P(A \cup B) = \frac{N_{A+B}}{N} = \frac{N_A}{N} + \frac{N_B}{N} = P(A) + P(B) $$.

# Expectation

Given random variable x with Pr(x) and a function f(x), we can calculate the expected value of f(x):

$$ E{[f{[x]}]} = \sum_x f(x) Pr(x) $$ for discrete x

$$ E{[f{[x]}]} = \int f(x) Pr(x) dx $$ for continuous x

For multiple variables x and y:

$$ E{[f{[x,y]}]} = \int \int f(x,y) Pr(x,y) dx dy $$

When thinking of expectations, remember these rules:

- the expected value of a constant k with respect to random variable x is k itself: $$ E{[k]} = k $$

- the expected value of a constant k times a function x is k times the expected value of that function $$ E{[kf(x)]} = k E{[f(x)]} $$

- the expected value of the sum of two functions of x is the sum of each of those expected values: $$ E{[f(x)+g(x)]} = E{[f(x)]} + E{[g(x)]} $$

- the expected value of the product of two functions f(x) and g(y) is the product of the individual expected values if x and y are independent: $$ E{[f(x), g(y)]} = E{[f(x)]} E{[g(y)]} $$

The expectations also have special names for some functions. Let's call the mean of the random variable x to be $$ \mu_x $$, then $$ f(x) = (x-\mu_x)^2 $$ is called the variance. $$ f(x) = (x-\mu_x)^3 $$ is called the skew. $$ f(x) = (x-\mu_x)^4 $$ is called the kurtosis and $$ (x-\mu_x)(y-\mu_y) $$ is called the covariance of x and y. 

We denote $$ m_n = E{[x^n]} = \int_{-\infty}^{\infty} x^n f(x) dx $$ to be the moments of the random variables x. The central moment is the mean $$ \mu_n = E{[(x-\mu)^n]} = \int_{-\infty}^{\infty} (x-\mu)^n f(x) dx $$

The absolute moment is $$ E{[\mid x \mid^n]} = E{[\mid x - \mu \mid^n]} $$

## Variance

The variance of f(x) is defined by:

$$ var(f) = E{[ (f(x) - E{[f(x)]})^2 ]} $$

This is how much variability there is in f(x) around its mean value $$ E{[f(x)]} $$. It is equivalent to:

$$ var(f) = E{[f(x)^2]} - E{[f(x)]}^2 $$

For one variable x, $$ var(x) = E{[x^2]} - E{[x]}^2 $$. For two random variables x and y, the covariance if defined by $$ cov(x,y) = E_{x,y} {[(x-E{[x]})(y-E{[y]})]} = E_{x,y} {[xy]} - E{[x]} E{[y]} $$ The covariance is to measure how much x and y vary together. If x and y are independent then the covariance is 0.