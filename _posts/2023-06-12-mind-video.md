---
layout: post
title:  "News: Reconstruct mind videos from brain activity"
date:   2023-06-12 10:14:54 +0700
categories: MachineLearning
---

# Introduction

Understanding the activities in our brain is a daunting task in neuroscience. So far success has been established for reconstructing static images from brain recordings. And recently, with the advancement of artificial intelligence (especially in deep learning), it has become possible to translate brain signals into semantically sensible video. This video interpretation from the brain is in line with our human internal visual experience, as we see life as if it is a continuous experience of a movie.

A common way to record brain activities is to use the fMRI (functional Magnetic Resonance Imaging) to measure blood oxygenation level dependent (BOLD) signals every few seconds. Technically, when a visual stimulus is presented to the brain, there would be dalays in BOLD signals. The response also varies across subjects and brain regions. This makes going from getting image representation to video recording from brain signals challenging. 

In "Cinematic Mindscapes: High quality video reconstruction from brain activity", the authors present MinD-Video, a model with two modules. The first module is called an fMRI encoder. It learns from brain signals using unsupervised learning. This enable to module to learn general visual features of the fMRI. The module learns deeper semantic patterns with annotated dataset (the encoder learns in the constrastive language image pre training space). The second module is fine tuned with a stable diffusion model, to generate videos. The analysis of the visual cortex and higher cognitive networks suggests that the model is sensible.

Let's explore the topic gradually with each building block.

# Image reconstruction

It has been shown that visual feature vectors (those computed by a convolutional neural network) can be predicted from fMRI patterns of the brain. Those predicted features can be used to identify the category of the object in the subject's head. To do that, we can benchmark the predicted feature vector to a set of computed features for numerous object images. Furthermore, it also demonstrates that object images in our head has high-level and low-level visual representations. This reinforces the fact that the CNN structure we usually use in machine vision is biologically sound.

Typically, a statistical classifier (decoder) would be trained to learn a relationship between fMRI patterns and the target content. The target could be the classes of the objects, or is expanded to be the image-level broader features. To capture a generic way of classifying objects, it is assumed that an object category can be represented by a set of visual features with invariances. Then using those visual features, an approach called generic object decoding is developed, to decode object categories from brain activity. To do so, a regression model (decoder) is trained to predict those visual feature patterns extracted from brain fMRI scans during viewing of those images. The predicted feature vector then is compared to the category-average feature vector of images in the database.

<img width="1221" alt="Screenshot 2023-06-12 at 13 26 49" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/2504a04d-59a6-44ca-af44-bff69de0c197">

In the image, the picture a shows how an image is read by each layer of the convolutional network. The CNN used has 8 layers: 5 convolutional layers and 3 fully connected layers. The other models are HMAX, GIST, and SIFT. Picture b shows the experiment process: the image is shown to the subject, then fMRI activities of the brain is recorded. Then the activities go through a decoder that was trained on a large dataset of images, brain activities and features pattern. The decoder predicts a pattern of the feature vector of the image. Then that pattern is compared with category average pattern set to identify most similar category. 

The similarity between the CNN architecture and how the brain actually works is also demonstrated in the following image:

<img width="629" alt="Screenshot 2023-06-12 at 13 35 54" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/4b4c94d1-8666-4063-8972-36c413f32483">

The image show the example of a feature neural unit (Unit 562 of layer 8 of CNN). The graph shows the true and predicted feature values for 50 test images predicted by the unit and from the whole visual cortex.


Another group of researchers have used similar approach: they train a model to learn the representation of fMRI data in a self supervised learning style. Then they use a latent diffusion model to reconstruct semantically sensible images. The self supervised learning task (Masked signal modeling) is a pretext task. A pretext task is a task that is not of interest on its own but it benefits downstream task in multiple ways. The pretext task here is to patchify and randomly mask the fMRI, then an auto encoder is trained to recover the masked patches. In doing this, the autoencoder learns the internal representation of the fMRI:

<img width="565" alt="Screenshot 2023-06-12 at 14 30 22" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/534631f8-d219-4ec1-9cfd-2de324469ccf">

In the second step, a diffusion model is used. In its basic form, a diffusion model is a probabilistic model defined by a two way Markov chain of states. There are two processes in the chain: the forward diffusion process in which noise is gradually added into the original image until the image disappears, and the reverse process in which the corrupted data would be recovered. A UNET architecture with attention mechanism (with key/value/query vectors) is used to condition image generation.

<img width="1172" alt="Screenshot 2023-06-12 at 14 23 10" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/a8ab3fec-15d2-4a9b-b597-7087cdabe4d0">

The latent diffusion model is emphasized to keep generation consistency so that images generated from similar brain activities should look semantically similar.

<img width="578" alt="Screenshot 2023-06-12 at 14 35 09" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/b7693fc0-4dff-4535-ba1b-2e17a9cf7a53">

# Video reconstruction

The conventional approach is to see video as a series of images. Those images have visual features, spanning many levels of abstration, such as orientation and color in the low level, shapes and textures in the middle levels and objects and actions in the high level. In the machine vision field, the convolutional neural network mimics the feedforward visual cortical network. Its work is similar to the brain's responses to natural image stimuli. To address the response of the brain to video stimuli, a group of researchers have acquired 11.5 hours of fMRI data from human subjects watching 1000 video clips. The videos show people in action, moving animals, nature scences, outdoor or indoor scenes. The CNN that is used is AlexNet. It has been pretrained. AlexNet has eight layers: the first five are convolutional layers, the last three are fully connected layers. The final layer is a softmax classifier, with output to be the normalized probabilities q. This is the probability that the image would be classified into categories. The network is trained using mini batch gradient descend to minimize the KL divergence from the predicted probability q to the ground truth p. The KL divergence is the amount of information lost when the predicted probability q is used to approximate p. The predicted probability $$ q = \frac{exp(yW + b)}{\sum exp(yW + b)} $$. The objective function is $$ D_{KL} (p \mid \mid q) = H(p,q) - H(p) = -(p . log q) + (p . log p) $$. H(p) is the entropy of p and H(p,q) is the cross entropy of p and q. The objective function is minimized with L2-norm regularization. Once trained, when we pass a frame of a movie through the CNN we would have an activation time series for each neural unit. A deconvolutional neural network (De-CNN) can be used to reconstruct the original image from the output. Specifically, the output can be unpooled, rectified and filtered until it reaches the pixel space. 

The cortical activations then are mapped with natural movie stimuli. Each movie segment was presented twice to each subject. This is to measure the intra-subject difference in voxel time series, since the way the brain reacts to stimuli are different across individuals. Then CNN unit outputs are log transformed and lag adjusted, then compared to the fMRI signals at cortical voxels. This study confirms the fact that the feedforward visual processing passes through multiple stages in both the CNN and the visual cortex. The authors then caclculate the correlations between the fMRI signal at each cortical location and the activation from CNN's layer. For each cortical location, the best corresponding layer is assigned an index. The layer index assignment provides a map of the feedforward hierarchical organization of the visual system. The authors also do voxel-wise encoding models, in which the voxel response is predicted from a linear combination of the feature representations of the input movie. First, they apply PCA to the feature representations from each layer of the CNN. PCA is to keep 99% of the variance while reducing the dimensions of the feature space. After PCA, the feature time series (since the CNN was shown a series of frames - as video) would be processed to match the sampling rate of fMRI. These feature time series then can be used to predict the fMRI signal at each voxel through a linear regression model

$$ x = Y.w + b + \epsilon $$

A L2-norm regularization would be applied:

$$ f(w) = \mid\mid x - Y.w - b \mid\mid_2^2 + \lambda \mid\mid w\mid\mid_2^2 $$

The L2-norm regularization is used to prevent overfitting. The regularization is optimized through a nine fold cross validation. 

<img width="889" alt="Screenshot 2023-06-12 at 16 24 40" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/58a679a0-43da-4120-bf38-5a877e163613">

Low level image features and classes then can be decoded from fMRI scans with video stimulus. Another group of researchers use conditional video GAN and generate better quality videos. The result is limited since GAN usually requires lots of data.

There is another approach using encoder-decoder architecture and self supervised learning. The authors train two separate networks: an encoder to map videos into fMRI samples, and a decoder to map fMRI samples to video frames. 

<img width="967" alt="Screenshot 2023-06-12 at 17 23 22" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/87879f7f-f6c8-4d77-8067-5c927ab90235">

The process is described in the image above, first the encoder is trained to translate video input into fMRI signals, then the decoder is trained to translate fMRI signals into video frames. Then there is a self supervised training process that train to keep the consistency between input video and output video, skipping the fMRI samples. The loss function for the encoder is:

$$ L_E(r,\hat{r}) = \mid\mid r - \hat{r} \mid\mid_2 + \alpha cos(r, \hat{r}) $$

with r, $$ \hat{r} $$ being the fMRI recording and its prediction, $$ \alpha = 0.5 $$. THe inital learning rate is 1e-4 and it is reduced every 3 epochs by a factor of 0.2. The encoder was trained for 10 epochs. The decoder loss is a bit more complicated:

$$ L_{singleFrame}(x,\hat{x}) = \beta L_{im}(x,\hat{x}) + \gamma L_{vid} (x, \hat{x}) + \delta L_R(\hat{x}) $$

with x, $$ \hat{x} $$ being the ground truth and the reconstructed frame. $$ L_{im} $$ is the similarity loss. $$ L_{vid} $$ minimize the L2 distance between embeddings of the ground truth frame and the reconstructed frame. $$ L_R $$ is a regularization term to encourage smoothness. $$ \beta = 0,35, \gamma = 0.35, \delta = 0.3 $$. 

As you can see, the resulting video has low semantical meaning.

# Masked braining model (MBM)

This is a pretext task that masks the fMRI and then let the autoencoder learns to fill in those masks. The knowledge learned by the autoencoder will transfer downstream.

# Contrastive language-image pretraining (CLIP)

CLIP is a pretraining technique that builds a shared latent space for images and natural languages by contrastive learning. The training will minimize cosin distance of paired image and text latent. The CLIP space will end up having rich semantic information on both images and texts.

# Stable Diffusion
Stable diffusion generates a latent version of the data instead of the data directly. As it works with the latent space, the computing resource is reduced and images with higher quality and better details can be generated.


```python

```
