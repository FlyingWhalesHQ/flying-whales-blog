---
layout: post
title:  "Stable Diffusion"
date:   2023-06-13 10:14:54 +0700
categories: DeepLearning
---

# Introduction

A diffusion process is a physical process in which a liquid gradually diffuses into another liquid, for example, milk particles into caffe. This idea is applied into a deep learning model called stable diffusion, to generate new images. In this model, there are two different processes: diffusion process and denoising process. In the diffusion process, the model starts with original images. It gradually adds Gaussian noise into the image in a series of time steps. This slow process chips away the original image's structure and detail, turning it into a simpler distribution. At the end, the image becomes completely noise (following a simple Gaussian distribution). The denoising process is the reversed process and it is used to generate new samples. It starts from samples from the Gaussian distribution (noisy images). Then it uses a neural network to trace out a reverse path. The path is to gradually remove the noise that was supposedly added during the diffusion process. If successful, this denoising process would learn to transform the Gaussian blob back into the structured piece of data. And then that same neural network can be used to generate a new image from a random noisy images. The generated image would look real, since the neural network has learned the underlying patterns of a real image, even if it was actually hallucinated. The training of the neural network involves comparing the denoised data to the original data at each step, and adjusting the network's parameters to minimize the difference.

# Diffusion model

Given a data distribution $$ x_0 \sim q(x_0) $$, a forward noising process q that adds Gaussian noise to x at time t with variance $$ \beta_t \in (0,1) $$ would be defined as follows:

$$ q(x_1, ...,x_T \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1} ) $$

$$ q(x_t \mid x_{t-1}) = N(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I) $$ 

The first step is to generate a sequence of beta value in each time step t. This sequence is a function of time t and is the variance of the Gaussian noise that would be added to the data at each time step t. Given a large T and a good schedule of $$ \beta_t $$, the latent data at timestep T would be completely noise, i.e. $$ x_T $$ is an isotropic Gaussian distribution. 

After having $$ \beta_t $$ we scale the data $$ x_{t-1} $$ by the factor $$ \sqrt{1 - \beta_t} $$ and the noise by factor $$ \beta $$. Then we sample for the $$ x_t $$. As the $$ \beta_t $$ increases, the image loses more of its structure.

Each step from $$ x_{t-1} $$ to $$ x_t $$ is a probabilistic one so the entire diffusion process can be viewed as a Markov chain. That is, a sequence of random variables in which the value of each step depends only on that of the previous step.

If we know the reverse distribution $$ q(x_{t-1} \mid x_t) $$ we can sample $$ x_T \sim N(0,I) $$ and run the process backward to get a sample from $$ q(x_0) $$. The distribution $$ q(x_{t-1} \mid x_t) $$ can be approximated with a neural network:

$$ p_{\theta} (x_{t-1} \mid x_t) = N(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)) $$

To denoise, first we define $$ \alpha_t = 1 - \beta_t $$ and $$ \bar{\alpha_t} = \prod_{s=0}^t \alpha_s $$, then we can define the sample of arbitrary step of the noised latent, conditioning on the original input $$ x_0 $$:

$$ q(x_t\mid x_0) = N (x_t; \sqrt{\bar{\alpha_t}}x_0, (1 - \bar{\alpha}_t) I) $$ and $$ x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon $$ where $$ \epsilon \sim N(0,1) $$.

Second, we can calculate the backward process as follows:

$$ \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t $$

$$ \tilde{\mu}_t(x_t,x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t  $$

$$ q(x_{t-1} \mid x_t, x_0) = N(x_{t-1};\tilde{\mu}(x_t,x_0), \tilde{\beta}_t I) $$

So the beta sequence is reversed, from high beta values to lower beta values. Then neural network is trained to maximize the log likelihood of the training data under the model. Once the neural network is trained, it is used to generate new samples. Starting from Gaussian noise, the denoising transformation is applied. The unstructured data gradually comes back into structured data, resembling the original data.

# Latent diffusion model

Since diffusion model typically operates directly in pixel space, the computation resource is quite expensive, usually takes hundreds of GPU days. The latent diffusion model was born to address this issue of computing resource while retaining the quality and flexibility of the model. In latent diffusion model, autoencoders are pretrained in the latent space. Since the computation is done mostly in the latent space which reduces the dimensions, a great amount of computation is reduced too.

There are two phases in the latent diffusion model. In the first one, an autoencoder is trained to provide lower-dimensional representational space. This space is more efficient since it has lower dimensions but has mostly the same amount of information. Then the diffusion model is trained in the learned latent space, for multiple tasks.

The adoption of such method offers many advantages over original diffusion models. Firstly, it increases computational efficiency. By changing the operation from high dimensional image space to lower dimensional latent space, diffusion models gain significant efficiency on computation. It enables the handling of larger dataset and more complex tasks. Secondly, it increases effectiveness with spatially structured data. Using excellent architecture such as UNET, the model can understand and utilize spatial relationship within the data, making it particularly well suited for image or spatial data processing. Thirdly, the model inherently is a general purpose compression scheme. It doesn't just have standard use cases but also is open to a wider range of applications. Since the latent space can be utilized to train multiple generative models, for multiple purposes, the utility and versatility is enhanced. Forthly, more downstream tasks become available, such as single image CLIP-guided synthesis. The output of this system can be fed into another system for further processing or analysis, increasing the complexity of the machine learning workflow.

The perceptual compression is an autoencoder trained by perceptual loss and patch based adversarial objective. This enforces local realism and avoid blurriness. Given an image x in RGB space, the encoder E would encode x into a latent representation z = E(x), and the decoder D would reconstruct the image from the latent representation, $$ \tilde{x} = D(z) = D(E(x)) $$. The autoencoder now has access to an efficient, low dimensional latent space. Apart from the saving on computational resource, this is also better, since the latent space helps to focus more on the important semantics of the data. The objective function of a diffusion model is:

$$ L_{DM} = E_{x,\epsilon \sim N(0,1),t} {[ \mid\mid \epsilon - \epsilon_{\theta} (x_t, t) \mid\mid_2^2 ]} $$

The objective function of a latent diffusion model would become:

$$ L_{LDM} = E_{E(x), \epsilon \sim N(0,1),t} {[ \mid\mid \epsilon - \epsilon_{\theta}(z_t,t) \mid\mid_2^2 ]} $$

We can also do conditional generative $$ p(z\mid y) $$. So the denoising autoencoder would be conditional and we can control the synthesis process though input y such as text, maps, and other tasks. A more flexible way to condition the image generator is to add cross attention mechanism into the UNET architecture. To encode y (such as class label), we introduce encoder $$ \tau_{\theta} $$ that encode y into $$ \tau_{\theta} (y) $$. This value is then mapped into UNET via the cross attention layer: 

$$ Attention (Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}}).V $$

with $$ Q = W_Q \phi(z), K = W_K \tau(y), V = W_V . \tau(y) $$

The objective function for a conditional latent diffusion model (LDM) is:

$$ L_{LDM} = E_{E(x), y, \epsilon \sim N(0,1),t} {[ \mid\mid \epsilon - \epsilon_{\theta} (z_t, t, \tau_{\theta}(y)) \mid\mid_2^2 ]} $$

# Example

Here is some images in the fashion MNIST that can be used for the training of diffusion model.

![originals](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/64dad6e1-0d33-4b82-ad11-adc13103edda)

Here is some result from the generative module of the diffusion model.

![ddpm_generated_images_plot](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/3d0346b8-28c3-47e1-ab1a-d44beda163e1)

# Conclusion

In the realm of generative models, diffusion models represent a new frontier of research. They allow the synthesis of high fidelity data samples of images and text, while maintaining a probabilistic nature, for variety and creativity. The transition from high dimensional space to a more manageable latent space by latent diffusion models is a promising advancement. The boundary of generative models will continued to be pushed in the near future, since there are so many applications and downstream tasks that benefit from them. We would expect to see more and more innovative applications and advancements in this field.


```python

```
