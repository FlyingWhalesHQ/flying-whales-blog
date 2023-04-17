---
layout: post
title:  "Object Detection: YOLO"
date:   2023-04-17 10:14:54 +0700
categories: DeepLearning
---
# TOC

- [Introduction](#intro)
- [Sliding CNN](#sli)
- [Region based CNN](#rcnn)
- [Fully CNN](#full)
- [YOLO](#yolo)


# Introduction

Speaking of image recognition tasks, we can distinguish them based on the nature of the task. For example, image classification is a task that labels images. Object localization is to draw a bounding box around one or multiple objects in the image. Object detection is a more difficult task that combines both tasks: to draw a bounding box and then label it. This task is critical for a wide range of applications, including surveillance, autonomous vehicles, robotics, and medical imaging.

Localizing an object in an image can be a regression task. We can predict a bounding box around the object, for example the coordinates of the center, plus its height and width. The measures are normalized such that the coordinates are in the range of 0 and 1. Another common version is to predict the square root of the height and width rather, so that a 10-pixel error of large box would be penalized less than the same error for a smaller box.

Another important and related problem in computer vision is the segmentation problem. That includes object segmentation and instance segmentation. Object segmentation involves identifying the pixels that belong to an object in an image, while instance segmentation goes a step further and differentiates between multiple instances of the same object. For example, in object segmentation, the algorithm needs to distinguish human from car. In instance segmentation, the algorithm needs to distinguish different humans from each other.

## IoU

IoU stands for Intersection over Union, which is a measure of the overlap between two sets. In the context of object detection, IoU is often used to evaluate the accuracy of a model's predictions by comparing the predicted bounding boxes with the ground truth bounding boxes. IoU is calculated as the ratio of the intersection of the predicted and ground truth bounding boxes to the union of the same boxes. It is expressed as a value between 0 and 1, where a value of 1 indicates a perfect overlap between the two boxes, while a value of 0 indicates no overlap at all. It measures how much area of the prediction is correct, regarding the union.

$$ IoU = \frac{\text{Area of overlap}}{\text{Area of union}} $$

# Sliding CNN
For multiple objects, there is a sliding CNN approach to this by sliding a CNN across the image grid and make prediction at each step. The method has one drawback, it can detect an object multiple times with multiple bounding boxes. To remedy this, we can use non max suppression technique. We set some threshold for the object score and remove all the bounding boxes with less than that score. Find the highest object score bounding box, and remove all remaining bounding boxes that overlap it a lot (with an IoU bigger than 60%).

# Region based CNN

The paper "Rich feature hierarchies for accurate object detection and semantic segmentation" - 2014 proposes an object detection system consists of three modules. The first generates region proposals (around 2000 regions). The second module is a CNN that extracts a fixed length feature vector from each region. The third module is linear SVMs. The first module uses selective search. 

The selective search generates small region of interests, then recursively combine adjacent regions. It considers four types of similarity when combining the smaller segmentation into larger ones:

- Color similarity: a color histogram of 25 bins is calculated for each channel then concanated into a color vector of 75 dimensions. Color similarity of two regions is then the histogram intersection: $$ S_{color} (r_i, r_j) = \sum_{k=1}^{n} min(c_i^k, c_j^k) $$ with $$ c_i^k, c_j^k $$ to be the histogram value for kth bin.

- Texture similarity: We extract Gaussian derivatives for 8 orientations in each channel, then a 10 bin historgram for each orientation and color channel. Texture similarity of two regions is then the intersection of histograms $$ S_{texture}(r_i, r_j) = \sum_{k=1}^{n} min(t_i^k, t_j^k) $$ with $$ t_i^k $$ being the histogram value of kth bin in the resulting vector.

- Size similarity: This similarity encourages smaller regions to merge eraly. $$ S_{size} (r_i, r_j) = 1 - \frac{size(r_i) + size(r_j)}{size(im)} $$ with size(im) being the size of the image in pixels.

- Fill similarity: This measures how well two regions fit. $$ s_{fill}(r_i,r_j) = 1 - \frac{size(BB_{ij} - size(r_i) -size(r_j)}{size(im)} $$  with size(BB) is the bounding box around region i and j.

- The final similarity is a linear combination of those four similarities above: $$ s(r_i, r_j) = a.S_{color}(r_i,r_j) + b.S_{texture}(r_i,r_j) + c.S_{size}(r_i,r_j) + d.S_{fill}(r_i, r_j) $$

After the region proposals, the authors do feature extraction by extracting a 4096 dimensional feature vector from each region with a CNN of 5 convo layers and 2 dense layers, inputing 227 x 227 pixels. They warp all inputs into 227 x 227 size regardless of aspect ratio and original size.

# Fully CNN

Semantic segmentation is to classify each pixel in an image to its class of object. In a fully CNN, the dense layers at the top are replaced with CNN. An example of a CNN that can input 448 x 448 images and output 10 numbers:

- Number 0 to 4 are put through a softmax function and turned into the class probabilities

- Number 5 is sent through the sigmoid function and gives the object score

- Number 6 and 7 are the bounding box's center coordinates. They go through a sigmoid function to be scaled back into 0 and 1.

- Number 8 and 9 are the bounding box's height and width.

This CNN can be converted into a fully CNN with the last layer to be a CNN instead of a dense layer. It will output 8x8 predictions.

<img width="597" alt="Screen Shot 2023-04-17 at 20 19 34" src="https://user-images.githubusercontent.com/7457301/232496269-5c632b41-e4a0-489f-b3ee-298798c52401.png">


# You only look once (YOLO)

YOLO framé the object detection task as a regression problem to both draw the bounding box and assign class probability. The contribution lies in the fact that they use a single neural net and one evaluation for the entire process. YOLO is also extremely fast, and it can process images in real time.

For each image, it divides the image into a S x S grid and for each grid cell predicts B bounding boxes, confidence for those boxes and C class probabilities. At the end, the final predictions keep the realest bounding boxes and its classification. 

The network architecture is like the GoogLeNet model for image classification, with 24 convolutional layers and 2 fully connected layers. However, the design is different in that instead of the inception modules they use 1x1 reduction layer followed by 3x3 convo layer. 

<img width="798" alt="Screen Shot 2023-04-17 at 22 34 57" src="https://user-images.githubusercontent.com/7457301/232537171-16a00995-1566-40a3-a961-fde6b8fa5eb2.png">

The net is pretrained on the ImageNet 1000 class competition dataset. They use the first 20 convo layers followed by an average pooling layer and a dense layer to train for a week to achieve 88% on the ImageNet 2012 validation set. Then the model is converted to perform detection by adding 4 convo layers and 2 dense layers (with randomly initialized weights). They double the input size to 448 x 448. The final layer is to predict class probabilities and bounding box coordinates. The width and height of the box are normalized to between 0 and 1 as usual. Same for the box coordinate. They use linear activation for final layer and leaky relu for all other: 

$$ 
\phi(x) = 
  \begin{cases}
    x, & \text{if x > 0 } \\
    0.1x, & \text{ otherwise } 
  \end{cases}
$$

The model optimizes for the sum squared error of the output with higher weight on loss of bounding box coordinate prediction $$ \lambda_{coord} = 5 $$, and lesser weight on confidence prediction of boxes without objects $$ \lambda_{noobj} = 0.5 $$. Here is the full loss function:


$$ Loss = Loss_{localization} + Loss_{confidence} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} {[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]} + \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} {[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 ]} + \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} (C_i - \hat{C}_i)^2 + \sum_{i=0}^{S^2} 1_i^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2 $$

where $$ 1_i^{obj} $$ denotes whether object appears in cell i and $$ 1_{ij}^{obj} $$ denotes that the jth bounding box predictor in cell i is responsible for that prediction. We see that the loss function only penalizes classification if an object is in that grid cell. It also only penalizes bounding box coordinate if that predictor is responsible for the ground truth (highest IoU).

The network was trained for about 135 epochs. The learning rate schedule is as follows: during the first epochs, the rate raise slowly from $$ 10^{-3} $$ to $$ 10^{-2} $$.  Then they continue to train with $$ 10^{-2} $$ for 75 epochs, then $$ 10^{-3} $$ for 30 epochs and finally $$ 10^{-4} $$ for 30 epochs. To relieve overfitting, dropout and data augmentation are used. A dropout with 0.5 rate is used after first layer. For data augmentation, random scaling and translation up to 20% of the original image size are used. Also random exposure and saturation up to 1.5 times are used.

Output of YOLO is a vector with bounding box coordinates and scores of classes $$ y^T = {[p_0, (t_x, t_y, t_w, t_h), (p_1, p_2, ...,p_c)]} $$ with $$ p_0 $$ to be the probability that the object presents in the bounding box, $$ t_x, t_y $$ being the center and $$ t_w, t_h $$ being the width and height. $$ \vec{p} $$ is the probability distribution over classes.

# Code example


```python
!git clone https://github.com/ultralytics/yolov5  # clone
!pip install -r requirements.txt  # install
```


```python
!python detect.py --source ../object2.jpg  

```

    [34m[1mdetect: [0mweights=yolov5s.pt, source=../object2.jpg, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
    [31m[1mrequirements:[0m /Users/nguyenlinhchi/Documents/GitHub/flying-whales-blog/_notebooks/requirements.txt not found, check failed.
    YOLOv5 🚀 v7.0-145-g94714fe Python-3.9.13 torch-2.0.0 CPU
    
    Fusing layers... 
    [W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
    YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
    image 1/1 /Users/nguyenlinhchi/Documents/GitHub/flying-whales-blog/_notebooks/object2.jpg: 448x640 3 persons, 2 bowls, 250.1ms
    Speed: 2.5ms pre-process, 250.1ms inference, 12.1ms NMS per image at shape (1, 3, 640, 640)
    Results saved to [1mruns/detect/exp5[0m


![object2](https://user-images.githubusercontent.com/7457301/232545754-3b87841b-77ef-4188-bd84-e351e6c03c38.jpg)


```python

```
