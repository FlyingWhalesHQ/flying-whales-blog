---
layout: post
title:  "Deep Learning in Healthcare"
date:   2023-04-20 10:14:54 +0700
categories: DeepLearning
---

# Introduction

To learn about self supervised learning and some other various machine learning techniques, let's read some academic papers that make uses of those techniques, demonstrating them. Let's explore two main papers on the application of deep neural network in healthcare, one in breast cancer screening: Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening, and one in the Covid healthcare assistance: Covid-19 prognosis via self supervised representation learning and multi-image prediction (by Facebook). In the first paper, a deep neural net with two stages is trained to learn the global and local information of breast scans. This strategy allows the model to achieves a human competitive performance and the heatmaps indicating locations of findings. The authors propose a novel variant of a ResNet designed for medical imaging, with balance of depth and width. This allows the processing of large image and maintaining a reasonable amount of memory consumption. In the second paper, the researchers use a self supervised net and transformers to predict the need of Covid's intensive care, inputing X-ray chest scan (one image and multiple image inputs). The result is promising, in that it can aid the hospitals to be prepared and to allocate limited resources more efficiently.

# Application in breast cancer diagnose

## Dataset
The dataset includes 230000 digital screening mamography exams, with 1 million images from 141000 patients. Each exam has at least four images, corresponding to the four standard views in screening mammography: R-CC (right craniocaudal), L-CC (left craniocaudal), R-MLO (right mediolateral oblique), L-MLO (left mediolateral oblique). Here are some examples of those images: 

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/42/9055242/8861376/wu1-2945514-small.gif">

The author also relies on diagnose of malignant or benign from bipopsies. There are 5800 exams with at least one biopsy performed within 120 days of the screening mammogram. The biopsies confirm malignant for 985 (8.4%) breasts and benign for 5500 (47.6%) breasts. 234 (2%) breasts have both malignant and benign findings. Example of labelling by a radiologist, with red being malignant and green as benign:

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/42/9055242/8861376/wu2ab-2945514-small.gif">

## Multi task learning

Multi task learning is training a single neural network to do multiple tasks at the same time: to identify / classify multiple labels in one image. For example, for one single image, the algorithm would answer questions such as "is it a dress? is it jean? does it have green color? does it have red color?". The output for each task could be 0 or 1 (0 for no and 1 for yes). The total output would be a vector of 4 dimensions, 2 elements for product and 2 elements for colors. For example, a red dress would be [1 0 0 1]. In the paper, for each breast there are two binary labels: the absence/presence of malignant findings  and the absence/presence of benign findings. Since we have left and right breasts, each exam has a total of four binary labels. The predictions are denoted: $$ \hat{y}_{R,m}, \hat{y}_{L,m}, \hat{y}_{R,b}, \hat{y}_{L,b} $$. In which the prediction of benign findings serves as an auxiliary task to regularize the main task of predicting a malignant finding. Four input images are denoted $$ x_{R-CC}, x_{L-CC}, x_{R-MLO}, x_{L-MLO} $$. Each CC image is cropped at 2677x1942 and each MLO is cropped at 2974 x 1748.

## ResNet

They have for model architechture: the view wise model that concatenates L-CC and R-CC, and L-MLO and R-MLO. Then it makes prediction for CC and MLO, then average those. The image wise model makes four predictions independently and then averaged. The side wise model concatenate L-CC and L-MLO, then R-CC and R-MLO then predict for left and right breasts. The joint model concatenates all four views and jointly predict malignant and benign findings for both breasts.

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/42/9055242/8861376/wu5-2945514-small.gif">

All models use ResNet-22 layers to output a fixed dimension hidden represnetation for each view and then two dense layers to map the hidden representations to the output predictions. What ResNet does is that it adds an intermediate input to the output of a series of convolution blocks. 

<img src="https://datagen.kinsta.cloud/app/uploads/2022/07/image1-3-1024x418.png">

This layer is simply F(x) + x itself and it is called a residual block since the mapping H(x) - (x) = F(x). This helps in the problem of vanishing gradient problem and adds original information skipping layers.

The author also tie weights for L-CC and R-CC, as well as L-MLO and R-MLO. Since they tie weights, they flip the left breast images before feeding into the model, so that all the breasts are rightward oriented. Weight tying is simply using the same weight matrix, and it is used as a regularization, to combat overfitting.

For the architecture, the first layer has 16 channels, with five ResNet blocks, each block doubles the number of channels, resulting in the last layer having 256 channels. The model is trained using Adam optimization algorithm, with a learning rate of $$ 10^{-5} $$ and a mini batch of size 4. $$ L_2 $$ regularization is applied to the weights with a coeffient of $$ 10^{-4.5} $$. The model has 6 million trainable parameters. They early stopped the training when the average of AUC did not improve for 20 epochs. 

## AUC

ROC curve stands for receiver operating characteristic curve is a graph showing the perfomance of a classification model at different classification threshold, by plotting two parameters: true positive rate and false positive rate. True positive rate (TPR) is recall and it is equal to $$ TPR = \frac{TP}{TP+FN} $$, false positive rate is defined as $$ FPR = \frac{FP}{FP + TN} $$. 

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg">

AUC is the area under the ROC curve.

<img src="https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg">

AUC is preferred in some cases since it is scale invariant. It measures how well the predictions are. Plus it is also classification threshold invariant, it measures the quality of the model's predictions irrespective of the threshold.

## Loss function

The prediction for each target is:

$$ \hat{y}_{R,m}(x_{R-CC}, x_{L-CC}, x_{R-MLO}, x_{L-MLO} = \frac{1}{2} \hat{y}_{R,m}^{CC} (x_{R-CC}, x_{L0CC}) + \frac{1}{2} \hat{y}_{R,m}^{MLO} (x_{R-MLO}, x_{L-MLO}), $$

and the loss function using binary cross entropy is:

$$ L(y_{R,m}, y_{L,m}, y_{R,b}, y_{L,b}, x_{R-CC}, x_{L-CC}, x_{R-MLO}, x_{L-MLO} = l(y_{R,m}, \hat{y}_{R,m}^{CC} (x_{R-CC}, x_{L-CC})) + ... $$


They use predictions from radiologists to compare results. 14 readers each reading 740 exams (1480 breasts): 368 exams randomly selected fromt he biopsied and 372 randomly selected. Exams are shuffled and readers give points on a scale of 0% to 100% for each breasts. The model achieves an AUC of 0.876 and the readers achieve from 0.705 to 0.860. So the authors evaluate a hybrid model of human-machine, with the formular $$ \hat{y}_{hybrid} = \lambda \hat{y}_{radiologist} + (1 - \lambda \hat{y}_{model} $$. With $$ \lambda = 0.5 $$, the hybrid achieves an AUC of 0.891, higher than as they do individually. This suggest the use of machine learning as a tool to assist radiologists. 

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/42/9055242/8861376/wu9ab-2945514-small.gif">

Example of one prediction, the following image has {'benign': 0.06533989310264587, 'malignant': 0.005473622120916843}

![result](https://user-images.githubusercontent.com/7457301/233016859-23093f9d-dd9d-470d-8bee-ff9cacfba64a.png)

# Covid-19's prognosis

## Abstract

The paper uses a self supervised method based on the momentum contrast (MoCo). This method would embed the image into a latent vector so that similar images are close and different images would be far from each other. Apart from the usual prediction for deterioration, adverse events (including transferring to the intensive care, intubation or mortality), and oxygen requirement heavier than 6L per day (since this requires the necessary masking instead of just nasal ventilation). The author then proposes a transformer based architecture that process multiple images. This achieves level of experienced radiologists. The result proves that the model can aid hospitals in allocating limited resources for needy patients in advance during the difficult emergency of Covid-19 pandemic. 

## Momentum Contrast - MoCo

MoCo is a self supervised method that rely on contrastive losses. It achieves representations that are as good as classification with labels, but independent of labels or tasks. That's why it is called self supervised. It is also called a pretext task. A pretext task is a task that is done not for the task genuinely but for the sake of learning a good data structure. In this case, when the model learns its data structure (features) without the need of labelling from human teachers. Those learned features can be used for training a classifier with target data. The classifier in this case is the prediction of increased care for a Covid case. The good thing is, they train a MoCo on two large, public chest X-ray datasets to have a pre-trained model. The knowledge of chest X-ray is then transferred downstream, into a feature extractor for the Covid's X-ray. In the literature, there is evidence that MoCo pretraining can be useful in learning representations that help identifying clinical condition.

<img width="370" alt="Screen Shot 2023-04-20 at 15 39 02" src="https://user-images.githubusercontent.com/7457301/233310020-d025fe92-f4db-4871-ad2f-e5b1067c7417.png">

MoCo is a way to build large and consistent dictionaries with contrastive loss: the dictionary is a queue of data samples with the encoded representations of the current mini batch are enqueued, and the oldest are dequeued. The key encoder is slowly progressing so that it maintains consistency: a momentum based moving average of the query encoder. A query would match a key if they are encoded views of the same images. 

There are many ways to define a loss function, serving different purposes of ours. A way is to measure the difference between the prediction and a target, such as rebuilding the input pixels (as in auto encoder models) by L1 or L2 losses, or classifying the input into categories and measuring the cross entropy or margin based losses. In this paper, they do the contrastive loss. It measures the simmilarities of sample pairs in representation space. Consider an encoded query q and a set of encoded sample $$ \{k_0, k_1, k_2, .. \} $$ that are the keys of a dictionary. Assume that there is one key k for q. A contrastive loss would make sure that it is low when q is similar to its key k, and dissimilar to all other keys. We use dot product to measure similarity, a form of contrastive loss InfoNCE would be considered:

$$ L_q = -log \frac{exp(q.k / \tau)}{\sum_{i=0}^{K} exp(q.k_i / \tau)} $$

This loss is for a softmax based classifier tries to classify q as k. 

## Dataset

The two datasets that are used for training are MIMIC-CXR (having 377000 chest X-ray images of 227000 studies) and CheXpert (having 224000 chest radiographs of 65000 patients). To fine tune the model for Covid, they use 27000 Xrays from 5000 patients. There are two labels: adverse events, and increased oxygen requirement. An adverse event is any of the three events: transfer to the intensive care, intubation or mortality. Each image is labeled with whether the patient developed any adverse event within 24, 48, 72 or 96 hours. The increase of oxygen resources is when patient requires more than 6L of oxygen per day.

## DenseNet

A DenseNet is a dense convolutional network, which connects each layer to every other layer in a feed forward fashion. A traddional convolutional networks with L layers will have L connections, but for a dense convolutional network there would be $$ \frac{L(L+1)}{2} $$ direct connections. For each layer, the feature maps of all preceding layers would be used as inputs. These connections alleviate the vanishing gradient problem, send original information across layers and reduce the parameters.

<img width="443" alt="Screen Shot 2023-04-20 at 16 52 39" src="https://user-images.githubusercontent.com/7457301/233330494-e450b363-6445-4adf-abc6-5da85feba054.png">

The idea of creating shortcuts from early layers to later layers is a general one. Many nets have this characteristics: ResNet, FractalNet, Highway Net, etc. By connecting all layers, information (all information indeed since all the feature maps) are fed forward. DenseNet structure is evaluated on competitive benchmark datasets (CIFAR-10, CIFAR-100, and ImageNet) and achieve good results.

Consider an image input $$ x_0 $$ and a network of L layers, each layer is a nonlinear transformation $$ H $$. H is a composite function of operations such as Batch Normalization, ReLU (rectified linear units), Pooling, or Convolution. Batch normalization is a layer that is used to stabilize the neural net by re-centering (minus the mean) and re-scaling (by standard deviation). Mathematically, the layer lth would receive the feature maps of all preceding layers $$ x_0, ..x_{l-1} $$ as input: $$ x_l = H_l({[x_0, x_1, ...x_{l-1}]}) $$. 

## Training
The image would be mapped into a latent space via the neural network. The mechanism would be that the contrastive loss is minimized. Specifically, a base image is transformed into $$ x_k, x_q $$ (augmentation). $$ x_q $$ would be passed through an encoder network, while $$ x_k $$ is passed through a momentum encoder network. This generates representation $$ r_q, r_k $$ for the image. The goal of the loss function (mentioned above) is to ensure that $$ r_q, r_k $$ come from the same underlying image, despite augmentations. They also encode the time stamp at which the image was taken. The encoded time stamp and the image are concatenated and went through transformer and pooling, finally to a linear classifier. During the process, some images are dropped, as a regularizer.


<img width="824" alt="Screen Shot 2023-04-20 at 21 10 05" src="https://user-images.githubusercontent.com/7457301/233392614-5a46d29a-03ef-4547-b3ea-54bad6633420.png">

The result is comparable, sometimes stronger than radiologist's reading. This helps the hospitals in emergency cases of patients to expect and allocate resources.


```python

```
