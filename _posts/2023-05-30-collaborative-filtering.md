---
layout: post
title:  "Collaborative Filtering"
date:   2023-05-30 10:14:54 +0700
categories: MachineLearning
---

# TOC
- [Introduction](#intro)
- [Memory based](#mem)
    - [User based](#user)
    - [Item based](#item)
- [Model based](#model)
    - [SVD](#svd)
    - [PCA](#pca)
    - [NMF](#nmf)
- [Hybrid](#hyb)
- [Deep learning](#deep)
    - [Auto encoder](#auto)

# Introduction

Collaborative filtering is a technique that is mainly used in recommendation system to suggest user's preferences by collecting data from a similar group of users. It assumes that if those people have similar preference and decisions in the past, so will they in the near future. So the prediction can be used as recommendation for the user on the products. Use case: Amazon develops item-to-item collaborative filtering in their recommendation system.

# Memory based

The memory based technique is to find users or items similar to the current user in the database and calculate the new rating based in the ratings of similar users or items found.

## User based

Let's start with a user-item rating matrix. We have the colums to be users and rows to be items. For large companies such as Netflix, Amazon, etc, the matrix can be quite large with millions of users and items. And the preference of a user over items can be obtained implicitly such as when a user watch a movie, it is assumed that she likes it. Or when she visits the item multiple times and for long, it can also be graded that she likes it. One issue with this utility matrix is that the company might not be able to obtain all ratings to fill in this matrix since each user might rate a few items only. So the matrix is quite sparse. There are several options to fill in those missing values. First is to place 0 for all the missing values. This is not a good choice since 0 means the lowest rating possible. A better option is to replace N/A with the average. For example, if the lowest is 0 and highest possible is 5, we can use 2.5. However, this has some issue with different kind of users. For example, an easy reviewer might rate 5 stars for a likeable move and for movies she doesn't like, she rate 3. If we replace her N/A with 2.5, all those movies mean bad movies for her. But for a difficult reviewer, she might only give 3 stars for movies she really likes and 1 star for movies she doesn't like. If we replace all her N/A with 2.5, we are assuming that she likes all the rest. A more reasonale option is to use a different average which is the average of all the ratings she has done. This alleviates the problem of easy and difficult user mentioned above. After filling those numbers in, we can normalize the matrix by substrating each user's rating column by their own average. This effectively set all the missing values to 0. Then 0 would be the neutral value and a positive value would mean that that user likes that movie. Likewise, a negative value means that the user doesn't like that movie. Doing that would make the matrix being stored more efficiently. And since sparse matrices can fit better into memory, it is also better for when we do the calculation.

The next step is to calculate similarity. We can use Euclidean distance or cosine similarity method.

## Euclidean distance

The Euclidean distance between two points in the Euclidean space is the length of the line segment between the two. It is also called the Euclidean norm of the two vector difference:

$$ d(x,y) = \mid\mid p - q \mid\mid = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2} $$

<img src="https://d138zd1ktt9iqe.cloudfront.net/media/seo_landing_files/formula-of-euclidean-distance-1624039148.png">

Image: Euclidean distance between two datapoints



```python
import numpy as np

p1 = np.array([0,1,5,3,2,4])
p2 = np.array([1,3,5,4,1,2])

distance = np.linalg.norm(p1 - p2)
print(distance)
```

    3.3166247903554



## Cosine similarity
Cosine similarity method is to calculate the cosine of the angle between two vectors formed by the two datapoints: $$ cos(u_1, u_2) = \frac{u_1^T u_2}{\mid \mid u_1 \mid\mid_2 . \mid\mid u_2\mid\mid_2} $$. With $$ u_1, u_2 $$ being normalized vectors. The value of the similarity would conveniently be between -1 and 1. A value of 1 means perfectly similar (the angle between these two vectors is 0). A value of -1 means perfectly dissimilar (the two vectors are on opposite directions). This means that when the behavior of the two users are opposite, they are not similar. 

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200911171455/UntitledDiagram2.png">

Image: the angle between two vector formed by two datapoints




```python
import numpy as np

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

# example vectors
p1 = np.array([0,1,5,3,2,4])
p2 = np.array([1,3,5,4,1,2])

print(cosine_similarity(p1, p2))  
```

    0.900937462695559



Other similarity indices that can be used are Pearson correlation coefficient, Jaccard similarity, Spearman rank correlation and mean squared difference. The Pearson correlation coefficient is the linear relation between the two samples. It is the covariance of the two variables divided by the multiplication of the two standard deviations: $$ \rho_{X,Y} = \frac{cov(X,Y)}{\sigma_X \sigma_Y} $$. Spearman rank correlation is the Pearson correlation equation but for rank variables. Jaccard similarity is when we count the number of items in the intersection set of the two sets then divide it by the number of items in the union set of the two sets: $$ J(X,Y) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid} $$. Mean squared difference is a popular measure for the distance between points.



```python
def jaccard_similarity(setA, setB):
    intersection = len(setA & setB)
    union = len(setA | setB)
    return intersection / union

# example sets
p1 = np.array([0,1,5,3,2,4])
p2 = np.array([1,3,5,4,1,2])

print(jaccard_similarity(p1,p2))  
```

    1.0


Notice that the two vectors have the same ratings but in different order. In this case, the jaccard similarity shows the perfect similarity which is not reasonable. If you plan to try this similarity index, please do keep this in mind.




```python
from scipy.stats import spearmanr

# Example data
p1 = np.array([0,1,5,3,2,4])
p2 = np.array([1,3,5,4,1,2])

correlation, p_value = spearmanr(p1, p2)

print("Spearman Rank Correlation: ", correlation)

```

    Spearman Rank Correlation:  0.6667366910003157



After establishing a way to calculate similarity, we then can move on to predict the rating. To calculate the rating prediction of user U for item I: we choose a list of top 5-10 most similar users who already rated I, then we take average of those ratings: $$ R_U = \frac{\sum_{u=1}^n R_u}{n} $$. This equation takes all the top similar users equally. Another choice is that we can weigh them accordingly: $$ R_U = \frac{\sum_{u=1}^n R_u * W_u}{\sum_{u=1}^n W_u} $$. The weights can be the similarity index for each user. Overall, this method is like asking your friends' suggestion when you want to see a new movie. We know that our friends have similar taste to us and so we can trust them. Their recommendations would be a reliable source of information.

### Code example
In this example, with the same movie lens dataset, we are going to do collaborative filtering: we create a user-item matrix (the utility matrix). Then we substract each rating with the average of rating of that user (this technique is to center the ratings). Then we fill N/A with 0. The next step is to calculate the cosine similarity for all users. By doing this, the system can list out the most similar users for a given user. Then if we need rating for a movie, we calculate the weighted average of the ratings by those most similar users.


```python
import pandas as pd
ratings = pd.read_csv('ml-latest-small/ratings.csv', low_memory=False)
ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.apply(lambda row: row - row.mean(), axis=1)
user_movie_matrix.fillna(0, inplace=True)
user_movie_matrix.head()
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
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.366379</td>
      <td>0.0</td>
      <td>-0.366379</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.366379</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9724 columns</p>
</div>




```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(user_movie_matrix, user_movie_matrix)
user_similarity_df = pd.DataFrame(cosine_sim)
user_similarity_df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>600</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.001265</td>
      <td>0.000553</td>
      <td>0.048419</td>
      <td>0.021847</td>
      <td>-0.045497</td>
      <td>-0.006200</td>
      <td>0.047013</td>
      <td>0.019510</td>
      <td>-0.008754</td>
      <td>...</td>
      <td>0.018127</td>
      <td>-0.017172</td>
      <td>-0.015221</td>
      <td>-0.037059</td>
      <td>-0.029121</td>
      <td>0.012016</td>
      <td>0.055261</td>
      <td>0.075224</td>
      <td>-0.025713</td>
      <td>0.010932</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001265</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-0.017164</td>
      <td>0.021796</td>
      <td>-0.021051</td>
      <td>-0.011114</td>
      <td>-0.048085</td>
      <td>0.000000</td>
      <td>0.003012</td>
      <td>...</td>
      <td>-0.050551</td>
      <td>-0.031581</td>
      <td>-0.001688</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.006226</td>
      <td>-0.020504</td>
      <td>-0.006001</td>
      <td>-0.060091</td>
      <td>0.024999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000553</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-0.011260</td>
      <td>-0.031539</td>
      <td>0.004800</td>
      <td>0.000000</td>
      <td>-0.032471</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.004904</td>
      <td>-0.016117</td>
      <td>0.017749</td>
      <td>0.000000</td>
      <td>-0.001431</td>
      <td>-0.037289</td>
      <td>-0.007789</td>
      <td>-0.013001</td>
      <td>0.000000</td>
      <td>0.019550</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.048419</td>
      <td>-0.017164</td>
      <td>-0.011260</td>
      <td>1.000000</td>
      <td>-0.029620</td>
      <td>0.013956</td>
      <td>0.058091</td>
      <td>0.002065</td>
      <td>-0.005874</td>
      <td>0.051590</td>
      <td>...</td>
      <td>-0.037687</td>
      <td>0.063122</td>
      <td>0.027640</td>
      <td>-0.013782</td>
      <td>0.040037</td>
      <td>0.020590</td>
      <td>0.014628</td>
      <td>-0.037569</td>
      <td>-0.017884</td>
      <td>-0.000995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.021847</td>
      <td>0.021796</td>
      <td>-0.031539</td>
      <td>-0.029620</td>
      <td>1.000000</td>
      <td>0.009111</td>
      <td>0.010117</td>
      <td>-0.012284</td>
      <td>0.000000</td>
      <td>-0.033165</td>
      <td>...</td>
      <td>0.015964</td>
      <td>0.012427</td>
      <td>0.027076</td>
      <td>0.012461</td>
      <td>-0.036272</td>
      <td>0.026319</td>
      <td>0.031896</td>
      <td>-0.001751</td>
      <td>0.093829</td>
      <td>-0.000278</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>605</th>
      <td>0.012016</td>
      <td>0.006226</td>
      <td>-0.037289</td>
      <td>0.020590</td>
      <td>0.026319</td>
      <td>-0.009137</td>
      <td>0.028326</td>
      <td>0.022277</td>
      <td>0.031633</td>
      <td>-0.039946</td>
      <td>...</td>
      <td>0.053683</td>
      <td>0.016384</td>
      <td>0.098011</td>
      <td>0.061078</td>
      <td>0.019678</td>
      <td>1.000000</td>
      <td>0.017927</td>
      <td>0.056676</td>
      <td>0.038422</td>
      <td>0.075464</td>
    </tr>
    <tr>
      <th>606</th>
      <td>0.055261</td>
      <td>-0.020504</td>
      <td>-0.007789</td>
      <td>0.014628</td>
      <td>0.031896</td>
      <td>0.045501</td>
      <td>0.030981</td>
      <td>0.048822</td>
      <td>-0.012161</td>
      <td>-0.017656</td>
      <td>...</td>
      <td>0.049059</td>
      <td>0.038197</td>
      <td>0.049317</td>
      <td>0.002355</td>
      <td>-0.029381</td>
      <td>0.017927</td>
      <td>1.000000</td>
      <td>0.044514</td>
      <td>0.019049</td>
      <td>0.021860</td>
    </tr>
    <tr>
      <th>607</th>
      <td>0.075224</td>
      <td>-0.006001</td>
      <td>-0.013001</td>
      <td>-0.037569</td>
      <td>-0.001751</td>
      <td>0.021727</td>
      <td>0.028414</td>
      <td>0.071759</td>
      <td>0.032783</td>
      <td>-0.052000</td>
      <td>...</td>
      <td>0.069198</td>
      <td>0.051388</td>
      <td>0.012801</td>
      <td>0.006319</td>
      <td>-0.007978</td>
      <td>0.056676</td>
      <td>0.044514</td>
      <td>1.000000</td>
      <td>0.050714</td>
      <td>0.054454</td>
    </tr>
    <tr>
      <th>608</th>
      <td>-0.025713</td>
      <td>-0.060091</td>
      <td>0.000000</td>
      <td>-0.017884</td>
      <td>0.093829</td>
      <td>0.053017</td>
      <td>0.008754</td>
      <td>0.077180</td>
      <td>0.000000</td>
      <td>-0.040090</td>
      <td>...</td>
      <td>0.043465</td>
      <td>0.062400</td>
      <td>0.015334</td>
      <td>0.094038</td>
      <td>-0.054722</td>
      <td>0.038422</td>
      <td>0.019049</td>
      <td>0.050714</td>
      <td>1.000000</td>
      <td>-0.012471</td>
    </tr>
    <tr>
      <th>609</th>
      <td>0.010932</td>
      <td>0.024999</td>
      <td>0.019550</td>
      <td>-0.000995</td>
      <td>-0.000278</td>
      <td>0.009603</td>
      <td>0.068430</td>
      <td>0.017144</td>
      <td>0.051898</td>
      <td>-0.026004</td>
      <td>...</td>
      <td>0.021603</td>
      <td>0.030061</td>
      <td>0.051255</td>
      <td>0.015621</td>
      <td>0.069837</td>
      <td>0.075464</td>
      <td>0.021860</td>
      <td>0.054454</td>
      <td>-0.012471</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 610 columns</p>
</div>




```python
def most_similar_users(user_id, user_similarity_df, n=5):
    # Get the series of similarity scores for the given user
    user_similarity_series = user_similarity_df.loc[user_id]
    
    # Exclude the user's own id (a user is always most similar to themselves)
    user_similarity_series = user_similarity_series.drop([user_id])
    
    # Get the most similar users
    most_similar_users = user_similarity_series.nlargest(n)
    
    return most_similar_users

# Get the 5 users most similar to user 1

similar_users = most_similar_users(1, user_similarity_df)
print(similar_users)

```

    188    0.134677
    245    0.106334
    377    0.099583
    208    0.089539
    226    0.085428
    Name: 1, dtype: float64


Notice that 0.13 doesn't show really high similarity.


```python
def get_ratings_for_movie(movie_id, similar_users, ratings_df):
    # Get the ratings of the most similar users for the given movie
    ratings = user_movie_matrix.loc[similar_users.index, movie_id]
    
    return ratings

# Get their ratings for the target movie
new_ratings = get_ratings_for_movie(2, similar_users, ratings)

print(new_ratings)

```

    188    0.000000
    245    0.000000
    377    0.000000
    208    0.000000
    226   -0.476331
    Name: 2, dtype: float64



```python
import numpy as np
np.dot(similar_users, new_ratings)
```




    -0.04069219402243481



User 1 is not likely to like movie 2. Or we can say that she/he is likely to be neutral. Since 4 out of the most 5 similar users (her neighbours) hasn't watched the movie yet.

## Item based

In the case of user based rating matrix, the matrix is usually sparse since each user only rates a few products. When one user changes rating or rate more, the average rating of that user changes and the normalization needs to be redone, leading to the need to redo the rating matrix as well. An approach that Amazon proposes, is to calculate the similarity among items then we can predict the rating of a new item based on those already rated items of the same user. Usually, the number of items are smaller than the number of users. This makes similarity matrix smaller and easier for calculation. Pick out a list of top 5-10 most similar items to the current one, rated by the user. Take their average or weighted average to indicate its rating. This method is developed by Amazon and can be used when there are more users than items, since it is more stable and faster. It is not very suited for datasets with browsing or entertainment related items.

# Model based

First to reduce the large but sparse user item matrix by matrix factorization. Matrix factorization is to breaking down a large matrix into a product of smaller ones. A = X * Y. For example a matrix of m users * n items can be factorized into the product of user matrix X (m * p feature)s and item matrix Y (p * n features). P are called latent features which means they are hidden characteristics of the user and the items.

If we reduce the dimension p into k < p, though, we force the algorithm to choose k main hidden characteristics that describe the most of the data possible. Then we multiple X_k (m * k) with Y_K (k * n), we get a new A' that approximates the original A. But in this new A', the missing values are filled in. This is how recommendation system works (similar to PCA). A general idea of this technique is called bottlenecking, we bottleneck the model so that only principal information comes through. And we end up with the "true" pattern underlying the dataset.

## SVD 

One of the popular algorithm to factorize matrix into meaningful components is the singular value decomposition algorithm. This method was introduced at length in a previous post on LSD/LSA (laten semantic analysis) for document processing. It is to factorized the utility matrix into a multiplication of user and item matrix: $$ A = U \Sigma V^T $$

where U and V are orthogonal matrices, $$ \Sigma $$ being the diagonal matrix containing the singular values of A. The singular values in $$ \Sigma $$ are sorted in decreasing order and the number of non zero singular values indicates the rank of matrix A.



```python
from scipy.sparse.linalg import svds
import numpy as np
np.set_printoptions(suppress=True)

# A simple user-item matrix
A = np.array([
    [0,1,5,3,4,5],
    [1,3,5,4,0,3],
    [3,4,2,0,2,1],
    [2,2,1,0,2,3]
])

# Perform SVD
U, s, VT = np.linalg.svd(A)

# print("U:\n", U)
# print("S:\n", S)
# print("VT:\n", VT)

# Construct diagonal matrix in SVD
S = np.zeros(A.shape)
for i in range(min(A.shape)):
    S[i, i] = s[i]

# Reconstruct Original Matrix
A_reconstructed = np.dot(U, np.dot(S, VT))
print("A (Full SVD):\n", A_reconstructed)

# Perform Truncated SVD
k = 2  # number of singular values to keep

U_k = U[:, :k]
S_k = S[:k, :k]
VT_k = VT[:k, :]

# Reconstruct Matrix using Truncated SVD
A_approx = np.dot(U_k, np.dot(S_k, VT_k))
print("A' (Truncated SVD):\n", A_approx.round(2))

```

    A (Full SVD):
     [[0. 1. 5. 3. 4. 5.]
     [1. 3. 5. 4. 0. 3.]
     [3. 4. 2. 0. 2. 1.]
     [2. 2. 1. 0. 2. 3.]]
    A' (Truncated SVD):
     [[ 0.23  1.69  5.33  3.77  2.38  4.56]
     [ 0.73  2.    4.42  2.9   2.19  3.83]
     [ 3.11  3.84  1.62 -0.11  1.9   1.64]
     [ 1.9   2.56  1.8   0.52  1.51  1.69]]



## PCA
PCA also decomposes the matrix into smaller matrices (in this case with eigenvectors). PCA algorithm has been introduced in a previous post.



```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming you have a dataset X with n samples and m features
X = np.array([
    [0,1,5,3,2,4],
    [1,3,5,4,1,2],
    [3,4,2,0,0,1],
    [2,2,1,0,2,3]
])
# Standardize the features
X = StandardScaler().fit_transform(X)

# Create a PCA object
pca = PCA(n_components=2) # we are reducing the dimension to 2

# Fit and transform the data
X_pca = pca.fit_transform(X)

print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

```

    original shape:    (4, 6)
    transformed shape: (4, 2)



## NMF

Non negative matrix factorization is a technique that decomposes a non negative matrix into the product of two non negative matrices: $$ V \approx W * H $$ with the constraint that V,W,and H does not contain any negative elements. The factorization is performed using an iterative optimization algorithm that minimizes the difference between V and the product of W and H: $$ \mid\mid V - WH \mid\mid^2 = \sum (V - WH)^2 $$. After having the two smaller matrices, we can predict the missing values using matrix multiplication. 



```python
from sklearn.decomposition import NMF
import numpy as np

# Create a simple matrix
V = np.array([
    [0,1,5,3,2,4],
    [1,3,5,4,1,2],
    [3,4,2,0,0,1],
    [2,2,1,0,2,3]
])
# Create an NMF instance
nmf = NMF(n_components=2, init='random', random_state=0)

# Fit the model
W = nmf.fit_transform(V)
H = nmf.components_

print("W:\n", W)
print("H:\n", H.round(2))
print("Reconstruction:\n", np.dot(W, H).round(2))

```

    W:
     [[1.53156606 0.        ]
     [1.29777613 0.68617373]
     [0.         2.01368557]
     [0.25974709 1.23548283]]
    H:
     [[0.   0.89 3.29 2.39 1.1  2.1 ]
     [1.52 1.9  0.78 0.   0.3  0.75]]
    Reconstruction:
     [[0.   1.36 5.03 3.66 1.69 3.21]
     [1.04 2.46 4.8  3.1  1.64 3.24]
     [3.06 3.83 1.57 0.   0.61 1.51]
     [1.88 2.58 1.82 0.62 0.66 1.47]]



# Hybrid

A hybrid model is one that makes use of both methods (memory based and model based). This is to leverage the strengths and compensate for weaknesses of both methods. For example, a hybrid model might use the model based method to predict a rating for an item, and then use a memory based method to generate prediction from similar users or items. So that they can compare or combine those predictions in some ways. The hybrid model clearly enjoys the patterns learned by the model based method, at the same time can keep the personalization by the memory method.

# Deep learning 

## Auto encoder

An auto encoder would learn the internal structure of a dataset, with hidden features, and then make prediction when new datapoint come in. Since a neural network approximates a function, this function also does bottleneck: to force the data in some way so that only the most important information can come through. Then the resulting model can be used to predict the missing values. Auto encoder has been introduced at length in a previous post.
