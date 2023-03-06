---
layout: post
title:  "K-means clustering"
date:   2023-03-02 18:14:54 +0700
categories: jekyll update
---

K-means clustering is a simple and intuitive algorithm. The introduction would be short. In K-means clustering, we are given a dataset without labels but has an inherent structure:

<img src='https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png'>

<p  style="font-size:9px">Source: https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning</p>

We can assume that there are K clusters (also called centroids). At the beginning, we initialize those centroids $$ \mu $$ randomly. Then each step we do two sub-steps:

We divide the data points x into each of those clusters so that the total squared distance between points $$ x_{i} $$ and centroids $$ \mu_{i} $$ is minimal. 
Similarly, with those assigned points, we move the centroids so that the process results in minimizing the total squared distance between points and centroids. The best solution is to let the centroids be at the center of their points (i.e. each is the average of its points). 

Here is the loss function of k-means clustering algorithm:

$$ Loss(x, \mu) = \sum_{i=1}^{n}||x^{(i)}-\mu_{k}||^{2}_{2} = \mu_{i=1}^{n}(x^{(i)}- \mu_{j})^{2}  \forall j \in (1,k) $$

In the first step, we assign x to $$ \mu $$ by giving it label j and then minimize loss (similar to MSE in which it takes the error measured by Euclidean distance) with respect to the given centroids:

$$ x \in j \leftarrow arg min Loss(x, \mu) $$

Easily we say that the best assignment of label y is to let each x belong to the cluster that has the closest centroid.

In the second step, with those points classified into clusters (labels j all set), we choose the new centroids that minimizes loss function:

$$ \mu \leftarrow arg min_{\mu} Loss(x, \mu) $$

$$ \Leftrightarrow \frac{\partial}{\partial \mu_{j}} \sum_{i=1}^{n}(x^{(i)}- \mu_{j})^2 = 0  \forall j \in (1,k) $$

$$ \mu_{j} \leftarrow \frac{1}{|{x^{(i)} \in \mu_{j}}|} \sum_{i \in \mu_{j}} x^{(i)} $$

For the new clusters, the best is to let the centroid to be the average of all its points.

Optimal number of clusters

There is a method called elbow in which we choose the optimal number of clusters for our dataset by choosing the elbow of the loss curve:

<img src='https://www.scikit-yb.org/en/latest/_images/elbow-4.png'>

<p   style="font-size:9px"> Source: https://www.scikit-yb.org/en/latest/api/cluster/elbow.html </p>


```python
!pip install yellowbrick
```

    Requirement already satisfied: yellowbrick in /Users/nguyenlinhchi/opt/anaconda3/lib/python3.9/site-packages (1.5)
    ...
    Requirement already satisfied: six>=1.5 in /Users/nguyenlinhchi/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.16.0)



```python
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import matplotlib.pyplot as plt

```


```python
pic = plt.imread('forests.jpg')/255
print(pic.shape)
plt.imshow(pic)

```

    (987, 1872, 3)





    <matplotlib.image.AxesImage at 0x7fef181a8cd0>




    
![png](Kmeans_files/Kmeans_2_2.png)
    



```python
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape
# looking with our own eyes, there should be 3 clusters
# one for the white part (where we cut down tree and make infrastructure)
# one for the light green (where human has cut down the tree and use the land for civil purpose)
# one for the bold green (the remaining forest)
kmeans = KMeans(n_clusters=3, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)

```




    <matplotlib.image.AxesImage at 0x7fef180b4430>




    
![png](Kmeans_files/Kmeans_3_1.png)
    



```python
# when we use the elbow method, it says to choose 4 clusters
# but 4 is not necessary
visualizer = KElbowVisualizer(kmeans, k=(2,12))
visualizer.fit(pic_n)        # Fit the data to the visualizer
visualizer.show()  
```


    
![png](Kmeans_files/Kmeans_4_0.png)
    





    <AxesSubplot:title={'center':'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>




```python
kmeans.cluster_centers_
```




    array([[0.74138631, 0.70173583, 0.66735134],
           [0.35342198, 0.49499163, 0.3090334 ],
           [0.1056781 , 0.17483956, 0.12261974]])




```python
# let's count the number of pixels in each group
# it is not very clear which category for those labels
pd.Series(kmeans.labels_).value_counts()

```




    2    1168258
    1     408385
    0     271021
    dtype: int64




```python
# so we use the pixel value counts
# the last line has the boldest of the color -> this is the forest
pd.DataFrame(pic2show).value_counts()
```




    0         1         2       
    0.105678  0.174840  0.122620    1168258
    0.353422  0.494992  0.309033     408385
    0.741386  0.701736  0.667351     271021
    dtype: int64




```python
# let's calculate how much forest left from this satellite image
271021/(1168258+408385+271021)
```




    0.14668305492773578




```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

```


```python
california_housing = fetch_california_housing(as_frame=True)
california_housing.data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = california_housing.data
y = california_housing.target
# Scale the dataset, since it has wildly different ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# concatnate y so that we cluster including y
y_new=np.array(y).reshape(-1,1)
data=np.append(X_scaled, y_new, axis=1)
data=pd.DataFrame(data)

kmeans2 = KMeans(n_clusters=10, random_state=0).fit(data)
visualizer = KElbowVisualizer(kmeans2, k=(2,12))
visualizer.fit(X_scaled)        
visualizer.show()  

```


    
![png](Kmeans_files/Kmeans_11_0.png)
    





    <AxesSubplot:title={'center':'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>




```python
# try again with the suggestion of 6 clusters
kmeans2 = KMeans(n_clusters=6, random_state=0).fit(data)

```


```python
# the housing data now is labeled by 6 groups
# this histogram shows that the housing market is distributed with polarity
plt.hist(pd.DataFrame(kmeans2.labels_))
```




    (array([7.600e+01, 0.000e+00, 8.758e+03, 0.000e+00, 1.214e+03, 0.000e+00,
            3.386e+03, 0.000e+00, 1.000e+00, 7.205e+03]),
     array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ]),
     <BarContainer object of 10 artists>)

