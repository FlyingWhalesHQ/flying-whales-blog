---
layout: post
title:  "Autoencoders in molecular simulations"
date:   2023-04-10 10:14:54 +0700
categories: DeepLearning
---

# Introduction

Autoencoders is a neural network architecture in unsupervised learning that maps the input into itself. Say, we have a set of unlabeled training examples $$ \{ x^{(1)}, x^{(2)}, .. \} $$, an autoencoder aims for the transformation $$ \hat{y}^{(i)} = x^{(i)} $$. This sounds trivial at first, but for the purpose, we set the hidden layer of the autoencoder to be smaller than the input (or bigger). This bottleneck (limitation in representational power), during training, would force the neural net to learn patterns in the input so that it can compress into a latent distribution z and then reconstruct with least information lost possible. The hypothesis function is simple:

$$ h_{\theta} (x) \approx x $$

It is the identity function and the neural net would learn to approximate it. For example, we have input x to be the pixel intensity values from a 28x28 images (784 pixels) so n = 784, and there are 300 hidden units in the hidden layer. The output would have 784 units as well. Since there are only 300 units in the middle, it must try to learn the correlation among pixels (the structure) and reconstruct the 784 pixel input x from the structure. 

The space of compressed representation is called the latent space. The network has two part: an encoder to compress the input and a decoder to expand the pattern back into its orignial size and content. Doing the encoding is like doing a PCA: reduce the dimension of the input. Mathematically, the encoder is the matrix transformation U, we would have $$ \hat{x} = U . y $$ So $$ \hat{x} = U . U^T . x $$. So x goes through the transformation of $$ U^T $$ into y and then transformed by U back into x. During the training, the autoencoder learns to minimize the distance between input and output (itself): $$ min \mid x - \hat{x} \mid $$ which is $$ min_U \mid x - U^T . U . x \mid $$. Autoencoders can have different architectures, including simple feedforward networks, convolutional networks for images, recurrent networks for sequential data. Variations of autoencoders do exist, such as denoising autoencoders, sparse autoencoders, variational autoencoders (VAE), and adversarial autoencoders. 

Denosing autoencoders add noise to the input data during training and then learn to reconstruct the original data from the noisy inputs. This is to prevent the trivial solution of simply copying the input over (in case of overcomplete autoencoder: the hidden layer is bigger than input). This gives result in task such as to reduce noise of images. VAEs are probabilistic models that learn to generate new data points from the learned latent space. AAEs use adversarial training to learn the distribution of the input data and the latent space.

# Variational autoencoder

A variational autoencoder follows usual autoencoder architecture: with an encoder and a decoder. The layer in the middle, however, is probabilistic. Instead of produce a vector representation for the input, it provides a mean $$ \mu $$  and a standard deviation $$ \sigma $$ for a Gaussian distribution. The decoder samples the coding randomly from a Gaussian distribution with given mean and standard deviation and then decodes the vector as usual. This is possible since during training, the cost function pushes the encoding to morph from the latent space into a Gaussian distribution space. Then to generate new instance, we just need to sample a random point in the Gaussian distribution and decode it. 

The cost function has two parts: a usual part to reconstruct the input (MSE - mean squared error). A second part which is the Kullback-Leibler (KL) divergence between the Gaussian distribution and the actual distribution of the latent space.

The first part is to ensure that the output resembles the input:

$$ min \mid x - \hat{x} \mid^{2} $$

The second part is to min the distance between latent distribution z and a Gaussian distribution $$ (\mu, \sigma) $$ using the KL difference:

$$ KL(q(z \mid x) || N(\mu, \sigma)) $$

The resulting decoder can act as a generator: take a random vector in z (or Gaussian) and decode it as an image. Or we can do image interpolation: take two points in Gaussian space and use the decoder to decode images from one point to another.

# Code example



```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```


```python
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
```


```python
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean 

tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU

codings_size = 10

inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="relu")(Z)
Z = tf.keras.layers.Dense(100, activation="relu")(Z)
codings_mean = tf.keras.layers.Dense(codings_size)(Z)  # μ
codings_log_var = tf.keras.layers.Dense(codings_size)(Z)  # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(100, activation="relu")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="relu")(x)
x = tf.keras.layers.Dense(28 * 28)(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
    axis=-1)
variational_ae.add_loss(tf.reduce_mean(latent_loss) / 784.)

variational_ae.compile(loss="mse", optimizer="nadam")
history = variational_ae.fit(X_train, X_train, epochs=25, batch_size=128,
                             validation_data=(X_valid, X_valid))
```


```python
def plot_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")
plot_reconstructions(variational_ae)
plt.savefig('reconstructed2')
plt.show()
```

![reconstructed](https://user-images.githubusercontent.com/7457301/230882274-794a0c39-c474-4a60-9cf3-460121ce9eff.png)



```python
codings = np.zeros([7, codings_size])
codings[:, 3] = np.linspace(-0.8, 0.8, 7)  
images = variational_decoder(codings).numpy()
def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = images.squeeze(axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
plot_multiple_images(images)
plt.savefig('interpolation2')
plt.show()
```

We can use the decoder to interpolate points in the latent space, so that in input space, we see interpolation between objects:

![interpolation](https://user-images.githubusercontent.com/7457301/230882269-00fa231e-a634-4fb3-9049-4c5da264725f.png)


# VAE in molecular simulation
<img width="605" alt="Screen Shot 2023-04-10 at 17 35 23" src="https://user-images.githubusercontent.com/7457301/230885884-56fed235-d70d-43b9-804f-e60c54a457b2.png">

In physics, coarse-graining (CG) is the technique to simplify the particle representation by grouping selected atoms into pseudo-beads, this improves simulation computation a great deal, hence facilitate the innovation in studying chemical space (such as protein folding). When doing so, some of the information would be lost in the process. If we want to study interactions at atomic level, we have the opposite process called backmapping: restoring fine-grain (FG) coordinates from CG coordinates. We add lost atomistic details back into the CG representation. Some of the difficulties involve: the randomness of backmapping - many FG configurations can be mapped to the same CG representations, hence the reverse generative map is one-to-many; the geometric consistency requirement: the FG coordinates should abide by the CG geometry in the way that they can be reduced back to the CG. Also, the CG transformation is equivariant, the FG transformation should be equivariant as well; different mapping protocols: there are different dimensionality reduction techniques of CG, depending on tradeoff between accuracy and speed, hence there should be a general backmapping approach that is robust among CG mapping choices.

<img width="607" alt="Screen Shot 2023-04-10 at 17 49 07" src="https://user-images.githubusercontent.com/7457301/230887881-9dd37fa9-c05f-4643-8799-4e635cb779e2.png">
Image: CG conversion and back (FG)

Recently, researchers have made progress significantly in this task, by utilising machine learning (specifically CGVAE model - Coarse-Graining Variational Auto-Encoder) for generative purpose. This approach can approximate atomistic coordinates based on coarse grained structures pretty well. The mathematical problem is to model the conditional distribution $$ p(x\mid X) $$ let FG molecular structures to be x and CG structure distribution to be X. The probability distribution x is considered latent (hidden and can be approximated/accessible via a similar but attackable distribution). The molecular geometry is parametrized and the requirement of geometry equivariant is incorporated into the backmapping function. Those are novel improvements in the field.

Let's use some maths notation. The FG system as atomistic coordinates can be written as $$ x = \{x_i\}_{i=1}^{n} \in R^3 $$. The CG system is $$ X = \{X_i\}_{i=1}^{N} \in R^{Nx3} $$ where N < n. We have 2 sets $$ {[n]} $$ and $$ {[N]} $$ and CG of molecular simulation is the assignment $$ m: {[n]} \rightarrow {[N]} $$ which maps each FG atom i in [n] to CG atom I in [N]. Each atom i can be in at most one I. We say X = M . x with $$ M = \frac{w_i}{\sum w_j} $$ if i in its group and 0 otherwise. Here $$ w_i $$ is the projection weight of atom i. It can be the atomic mass i or 1, when it is at the center of mass of the group. CG is the weighted average of FG geometries.

The geometric requirement of backmapping is for the CG to follow the rules of FG in rotate and reflect. It is equivariant requirement. The backmap x can also be mapped back into X itself. Those requirments will be satisfied by the generative CF VAE framework. Since the probability $$ p(x\mid X) $$ is intractable, we can approximate a parameterized version of the function $$ p_{\theta}(x\mid X) $$, in our case, with a deep neural network.

Further work has adapted the VAE framework proposed above to build an impressive backmapping tool for CG protein representation. They achieve it through: base representation on internal coordinates, enforce an equivariant encoder, a custom loss function that ensures local structure, global structure and physical constraints. They also train the model on a high quality curated protein data.

The loss function is very similar to VAE: $$ Loss = L_{reconstruction} + \alpha L_{KL divergence} $$, only different in its qualitative meaning. They need to ensure validity of the generated structure (with geometry and interactions at atomic level), they supervise the model on both topology and atom placements. Topology reconstruction is measured by MSE loss on bond lengths and angular loss. 

$$ L_{bond} = \frac{1}{\mid B \mid} \sum(b - \hat{b})^2 $$

$$ L_{angle} = \frac{1}{\mid A \mid} \sum \sqrt{2(1-cos(\theta - \hat{\theta})) + \epsilon} $$

B is the set of all bonds, b is the ground truth and $$ \hat{b} $$ is the predicted bond. A is the set of all angles, $$ \theta $$ is the ground truth and $$ \hat{\theta} $$ is the predicted angle in radian.

They also add root mean squared distance (RMSD) loss for Catersian coordinate space:

$$ L_{torsion} = \frac{1}{\mid T \mid} \sum \sqrt{2 (1-cos(\phi - \hat{phi})) + epsilon} $$

$$ L_{coor} = \frac{1}{\mid N \mid} \sum \mid x - \hat{x} \mid^{2}_{2} $$

They also put constraints on the chemical validity of the structure, the following steric loss makes sure the distance between any two non bonded atom pairs larger than 2.0 A.

$$ L_{steric} = \sum \sum max(2 - \mid x - y \mid_{2}^{2} $$

In total, $$ L_{recon} = \gamma L_{local} + \delta L_{torsion} + \beta L_{coor} + \eta L_{steric} $$


```python

```