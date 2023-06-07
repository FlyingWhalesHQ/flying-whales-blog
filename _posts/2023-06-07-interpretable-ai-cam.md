---
layout: post
title:  "Interpretable AI: CAM"
date:   2023-06-07 10:14:54 +0700
categories: MachineLearning
---

# Introduction

CAM (Class Activation Mapping) is a way to understand how and why a deep learning model has arrived at its prediction. It adds transparency to the applications, especially helpful in healthcare, finance, and autonomous vehicles where safety, compliance and robustness are important. It does so by visualize the decision of a convolutional neural network on the image. Roughly speaking, it shows where the model was looking at while it made the decision. This means that CAM provides a spatial map of the important features/pixels for the task at hand, giving some explanation for it. It can easily imagined that a heatmap of where the decision was focused on would provide great aid to doctors in medical imaging tasks.

# CAM

CAM's authors argue that the convolutional units in the CNN are the part that actually localize objects in the images despite having not being instructed explicitly. This ability would be diluted in the last layer of fully connected neurons. To avoid this, some new network architecture was developed to be fully convolutional. Of those, some use a global average pooling layer, that acts as a structural regularizer, preventing overfitting. The authors provide some tweaking to make such network able to retain the ability to localize discriminative regions.

The authors use a similar network architecture to GoogLeNet, with mostly convolutional layers, and then just before the final softmax output, they put a global average pooling (GAP) as a fully connected layer. They would then project the weights of the fully connected layer back to the last convolutional feature maps (so it is called class activation mapping).

<img width="1217" alt="Screenshot 2023-06-07 at 14 14 26" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/2c55180f-428a-419d-8b79-666dbd3b8a0b">

Specifically, let the last convo layer produces K feature maps $$ A^k \in R^{uxw} $$ of width u and height v. These feature maps are then pooled using global average pooling method and linearly transformed to produce a sscore $$ y^c $$ for each class c:

$$ y_c = \sum_k w_k^c \frac{1}{Z} \sum_i \sum_j A_{ij}^k $$

The localized map $$ L_{CAM}^c \in R^{uxv} $$ for class c is the linear combination of the final feature maps with the learned weights of the final layer: $$ L_{CAM}^c = \sum_k w_k^c A^k $$. This would then be normalized between 0 and 1.

The authors also make a different between using global average pooling and global max pooling method. The global average pooling layer would encourage the network to recognize the whole extent of objects. Meanwhile the average max pooling only identify one discriminative part. 

# Code example

In this code example, you would find the implemenation of CAM and its heatmap on the original image. Since the image contains a dog and a cat, with the dog being more prominent, the model focuses on the head of the dog the most. This means that VGG16 is looking at the right place to base its decision on.


```python
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

# Load VGG16 model
model = VGG16(weights='imagenet')

# Load the image
img_path = '/kaggle/input/photo2monet-examples/cat-dog.jpg'  # Insert your image path here
original_img = cv2.imread(img_path)
width, height, _ = original_img.shape

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get the predictions
preds = model.predict(x)

# Take the topmost class index
top = np.argmax(preds[0])

# Take output from the final convolutional layer
output = model.output[:, top]
last_conv_layer = model.get_layer('block5_conv3')

# Compute the gradient of the class output value with respect to the feature map
grads = K.gradients(output, last_conv_layer.output)[0]

# Pool the gradients over all the axes leaving out the channel dimension
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# Weigh the output feature map with the computed gradient values
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):   # we have 512 features in our last conv layer
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Average the weighted feature map along the channel dimension resulting in a heat map of size 14x14 
heatmap = np.mean(conv_layer_output_value, axis=-1)

# Normalize the heat map to make the values between 0 and 1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Resize heatmap to original image size
heatmap = cv2.resize(heatmap, (height, width))

# Plot original image and heatmap side by side
plt.figure()
plt.imshow(original_img)
plt.title('Original Image')

plt.figure()
plt.imshow(heatmap)
plt.title('Class Activation Map')
plt.show()
```

![cat-dog](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/473b5b41-c0a4-4361-922d-fb27b6263f10)

![CAM](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/eae97d23-ac18-487d-b23d-90e8bbfd3015)

# GradCAM

GradCAM is part of the effort to make intelligent systems more trustworthy and easier to integrate into human's life. GradCAM, short for gradient weighted CAM, can visualize the heatmap for any layers inside the neural network, not just the final one. And it doesn't just supplies a way to build trust, it can also be used to compare between a stronger and weaker model when both models make the same predictions based on their different visualization. It provides a new way to understand CNN models.

To compute the class discriminative localized map grad-CAM $$ L_{Grad-CAM}^c \in R^{uxv} $$ in any CNN, we need to compute the gradient of $$ y^c $$ with respect to feature maps A of the interested convo layer: $$ \frac{\delta y^c}{\delta A_{ij}^k} $$. The weight $$ \alpha_k^c $$ is the partial linearization of the network from A, it is the importance of feature map k for class c. The grad-CAM heatmap is then a weighted combination of feature maps, but in ReLU way:

$$ L_{Grad-CAM}^c = ReLU (\sum_k \alpha_k^c A^k) $$

This results in a coarse heatmap, which is then normalized for visualization. 

<img width="801" alt="Screenshot 2023-06-07 at 15 17 32" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/668fae14-23f1-4f36-ab06-c696771e55a8">

# Example

For the neural network, you can use Xception, VGG16, ResNet50, etc. The results are similar, and we use the same image which is classified as "golden retriever". In this example, we will use Xception. Firstly, run `model.summary()` to print out the layers of the architecture. We will see the heatmap of the last convo layer "block14_sepconv2_act". It is similar to the VGG16's heatmap we saw above with CAM method.

![xception-lastlayer](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/35608aa2-403c-4660-a064-bef13ae2ab2e)

Let's overlay this heatmap over the original image:

![overlay-2](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/75a215f3-4edc-449b-a125-fa65e43bf9ba)

We can see that the model looks at only the dog's face, it decides to classify this as a dog.

Secondly, let's look at an inner convo layer of the model, to see what they see. This is the part that CAM cannot do. We will look at this layer called "block9_sepconv3_act".

![xception-midlayer](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/c44d2c6e-4e82-4a20-b72f-b39cddc5b831)

Let's superimpose this heatmap on the original image:

![cam](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/b716108d-c392-4c84-ab96-f5098c311c39)

We can see that this layer looks at the paws and the body parts of the two animals.

Lastly, if we pointwise multiply the gradCAM matrix and the guided backpropagation, we have a fusion of class discriminative and high resolution guidance called guided Grad-CAM:

![guided](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/43180825-ebf2-41d7-8e5a-e48a7b6369d3)

# Conclusion

In conclusion, the GradCAM is a powerful technique to understand the decision making process of convolutional neural networks. It is a generalization of CAM ($$ \alpha = w $$ at the last layer). It uses the gradients flowing backward into each layer to calculate the importance of each pixels and provide a coarse map highlighting the regions that the model uses for prediction.

GradCAM, together with other discussed techniques, have shedded much light into the blackbox as we know it. And this interpretability is vital for AI and machine learning system while they continue to evolve and fit into human society. We make them, we need to take the responsibility to keep them transparent, trustworthy and accountable. 

I believe that in the future, more and more contribution in the interpretability of AI will appear and be welcomed by people from across disciplines, not just the end users. Since interpreting AI models is undeniably an integrated part of progress.


```python

```
