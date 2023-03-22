---
layout: post
title:  "Modern CNNs"
date:   2023-03-09 11:14:54 +0700
categories: jekyll update
---
# TOC

- [Introduction](#intro)
- [LeNet](#lenet)
- [AlexNet](#alex)
- [GoogLeNet](#google)
- [ResNet](#res)

# Introduction

Convolutional Neural Network (CNN for short) is an important tool in modern computer vision. Today we will see the development of CNN over time, starting from a shallow net to very deep and complex (with internally intertwined) architecture.

# LeNet (1998)

LeNet was architected to recognize hand writings. Its CNN is the recognizer part of a hybrid model with four component: a word preprocesser that normalize word with EM algorithm, an encoder to turn images into annotated version, a recognizer which is the CNN, and the word scorer using a hidden Markov model. In this post, we care about the CNN only.

The CNN has five layers: first a convolution layer with eight kernels of 3x3. Kernal is called local receptive fields (a concept in neuroscience) since its dimensions are smaller than the input, when it slides over the image, it is as if the neural net is able to see a small window of the image at a time. So it is used to detect local feature. And each neuron in the next layer is connected to a local part of the image in the previous step. The second layer is a 2x2 subsampling (basically a pooling) layer. Layer 3 is another convolution with 25 kernels of size 5x5. Layer 4 is a convolution with 84 kernels of 4x4. Layer 5 is a 2x1 subsampling. And the classification layer is a RBF (radial basis function) with 95 units (one per class).

<img width="1036" alt="Screen Shot 2023-03-21 at 08 29 56" src="https://user-images.githubusercontent.com/7457301/226499201-2d6ce06e-7d3e-464f-8dfd-8bf0eeabcc1d.png">



```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D # new! from keras.layers import Flatten

model = Sequential()
# first convolutional layer:
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28, 28, 1)))
# second conv layer, with pooling and dropout:
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# dense hidden layer, with dropout:
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# output layer:
model.add(Dense(n_classes, activation='softmax'))
model.summary()

```

# AlexNet (2012)

AlexNet was trained on ImageNet repertoire - one of the biggest image dataset til date, and one of the most popular dataset to benchmark among neural network. It contains eight main layers: five convolutional layers and three fully connected layer, with the softmax of 1000 classes at the end for the classification task. One of the characteristics of the neural net is that it uses nonlinear ReLU (Rectified Linear Unit) $$ f(x) = max(0,x) $$ instead of tanh or sigmoid. This is not just computationally fast but it is also a non saturating neuron. Non saturating neuron is to combat gradient vanishing, a phenomemnon in which gradient simply goes to zero during complex calculation because of the calculation formula (for example, when x is very large, sigmoid reaches its limit zone). Without ReLU, we would not be able to explore large models. They also use a technique to normalize the data after ReLU called local response normalization and some specific pooling called overlapping pooling.

<img width="989" alt="Screen Shot 2023-03-21 at 08 33 46" src="https://user-images.githubusercontent.com/7457301/226499518-4ddb9ba1-ab93-4e51-8293-0cdd8e283529.png">

As in the picture, the first layer takes input of 224x224x3 of a RGB image, filtered it with 96 kernels of 11x11x3. Which is equivalent to multiply the image pixel matrix with a matrix of 11x11x3 and repeat that 96 times with 96 different kernel matrices. This creates 96 resulting matrices called feature map. We can see as if we are using 96 ways of looking at/transforming the original image, each way focus on one particular feature. For example, one kernel is responsible for looking only at horizontal line, it will interpret the image regarding its perspective of seeing only horizontal lines. The output from the first layer is then locally response normalized and overlappingly pooled. It becomes the input for the second convo layer. In the second convo layer, there are 256 filters (it is a good practice to double the filter in early layers) size 5x5x48. The third, fourth, and fifth layers don't have pooling or normalization but the third has 384 kernels of 3x3x256, the fourth has 384 kernels of size 3x3x192 and the fifth has 256 kernels of size 3x3x192. Each fully connected layer has 4096 neurons. Each hidden neuron in the first two fully connected layers has 0.5 probability to drop out (output goes to zero). Those would not feedforward anything and would not participate in the backpropagation process. They share weights though, so that others learn to do the job of those blanked out neurons. Dropout reduces overfitting but takes longer for the gradient to converge. They also used two ways of data augmentation: extracting random patches and doing PCA to alter the intensities of RGB pixels. Follows is an implementation of AlexNet:


```python

model = Sequential()
# first conv-pool block:
model.add(Conv2D(96, kernel_size=(11, 11),
          strides=(4, 4), activation='relu',
          input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# second conv-pool block:
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# third conv-pool block:
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# dense layers:
model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
# output layer:
model.add(Dense(17, activation='softmax'))

```

The training of AlexNet has become good practice in the industry: to optimize the multinomial logistic regression using mini batch gradient descent (with backpropagation). The batch size is 256, weight decay of $$ 5.10^{-4} $$ and drop out for the first two fully connected layers is 0.5. The learning rate was initially $$ 10^{-2} $$ then decreased by a factor of 10 when the validation set accuracy stopped improving. By that, the learning rate decreased 3 times during training. For data augmentation, the image was scaled, random horizontal flipped and RGB shifted. The score that was used was top-1 and top-5 error rate. For a top-5 error rate, it is the proportion of images that are incorrectly classified (the correct label is not in the top-5 outputed by the model).


# VGG Net (2014)

VGG net was also born for the ImageNet challenge, it inputs images of 224x224 RGB. It preprocesses by substracting the mean of pixel to each pixel. Then the input is passed through a stack of CNN, with filters of 3x3 and even 1x1 (linear transformation). The stride is 1 (when we slide the filter across the image, we can choose to jump 1 or 2 steps, this is the stride, 1 pixel striding is the smoothest sliding). Since use a 3x3 filter, the pixel in edges would not be seen, we make them be seen by padding 0 around the original image. Here they pad so that the spatial resolution is same after convolution. Then they do some spatial pooling with 2x2 max pooling, stride of 2. The stack of CNN is then followed by a FCN (fully connected neural layer): the first two have 4096 neurons, the third has to do a classification of 1000 classes hence it has 1000 neurons. The final layer turns those outputs into propensities by using softmax function. Other than the softmax layer, they use ReLU for all the activation.


<img width="743" alt="Screen Shot 2023-03-21 at 10 40 21" src="https://user-images.githubusercontent.com/7457301/226512464-bccfe5df-a98e-4891-b4ea-f9e29cde626e.png">


```python

model = Sequential()
model.add(Conv2D(64, 3, activation='relu',
          input_shape=(224, 224, 3)))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu'))
model.add(Conv2D(256, 3, activation='relu'))
model.add(Conv2D(256, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='softmax'))


```

The training is almost the same as the AlexNet's procedure, with the objective function, error rate, learning rate and such. It is trained on the ImageNet of 1000 classes, with 1.3 million images, 50 thousands images for validation and 100K for held out testing.

# GoogLeNet/Inception (2015)

The GoogLeNet was design with hardware implementation and real world application in mind. So they figured a way to reduce the parameters in the inception block.


<img width="431" alt="Screen Shot 2023-03-21 at 13 00 28" src="https://user-images.githubusercontent.com/7457301/226529312-af6267eb-885a-4db8-b6bc-93e5c44bb935.png">

<img width="364" alt="Screen Shot 2023-03-21 at 13 00 11" src="https://user-images.githubusercontent.com/7457301/226529318-b0f0350d-52d0-4386-909a-5cbb0759a92e.png">


All the convolution using ReLu activation. the input is 224x224 RGB substracted for the mean. Filters are mostly 3x3 and 5x5. An inception module let the input go through several convolution layer and then concatenate all of them together. The auxiliary layer (to the side) uses average pooling with 5x5 filter and stride 3, resulting in 4x4x512 output map; 1x1 convo with 128 filters to reduce the dimension and ReLU for nonlinear activated; a fully connected layer with 1024 unit, plus ReLU as usual; a dropout with 70%; a linear layer with softmax loss as the classifier.

# ResNet (2015)

ResNet stands for residual net, in which we skip connection and plus the input straight to end output of that block of neurons.

<img width="455" alt="Screen Shot 2023-03-21 at 13 11 09" src="https://user-images.githubusercontent.com/7457301/226530558-640cb53a-1f36-4cff-9a1c-cc7fb01a470c.png">

Despite adding the input to the end output, it is called residual for the following reason: denote an underlining mapping (function) to be H(x). We let the layer fit the residual H(x) - x. Name the residual F(x) = H(x) - x, we have the equivalence F(x) + x = H(x). Written in this form, we see that the end ouput is added a term x which is exactly the plain input (without transformation). The ended-up neural net represents the true underlining mapping H(x).



```python
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
model = tf.keras.Sequential([
    
DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))

```


## DenseNet

DenseNet is an extension of ResNet in which we add a lot of those skip connections as describe above.
