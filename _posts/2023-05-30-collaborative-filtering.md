---
layout: post
title:  "Collaborative Filtering"
date:   2023-05-30 10:14:54 +0700
categories: MachineLearning
---

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

## Cosine similarity
Cosine similarity method is to calculate the cosine of the angle between two vectors formed by the two datapoints: $$ cos(u_1, u_2) = \frac{u_1^T u_2}{\mid \mid u_1 \mid\mid_2 . \mid\mid u_2\mid\mid_2} $$. With $$ u_1, u_2 $$ being normalized vectors. The value of the similarity would conveniently be between -1 and 1. A value of 1 means perfectly similar (the angle between these two vectors is 0). A value of -1 means perfectly dissimilar (the two vectors are on opposite directions). This means that when the behavior of the two users are opposite, they are not similar. 

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200911171455/UntitledDiagram2.png">

Image: the angle between two vector formed by two datapoints

Other similarity indices that can be used are Pearson correlation coefficient, Jaccard similarity, Spearman rank correlation and mean squared difference. The Pearson correlation coefficient is the linear relation between the two samples. It is the covariance of the two variables divided by the multiplication of the two standard deviations: $$ \rho_{X,Y} = \frac{cov(X,Y)}{\sigma_X \sigma_Y} $$. Spearman rank correlation is the Pearson correlation equation but for rank variables. Jaccard similarity is when we count the number of items in the intersection set of the two sets then divide it by the number of items in the union set of the two sets: $$ J(X,Y) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid} $$. Mean squared difference is a popular measure for the distance between points.

After establishing a way to calculate similarity, we then can move on to predict the rating. To calculate the rating prediction of user U for item I: we choose a list of top 5-10 most similar users who already rated I, then we take average of those ratings: $$ R_U = \frac{\sum_{u=1}^n R_u}{n} $$. This equation takes all the top similar users equally. Another choice is that we can weigh them accordingly: $$ R_U = \frac{\sum_{u=1}^n R_u * W_u}{\sum_{u=1}^n W_u} $$. The weights can be the similarity index for each user. Overall, this method is like asking your friends' suggestion when you want to see a new movie. We know that our friends have similar taste to us and so we can trust them. Their recommendations would be a reliable source of information.

## Item based

In the case of user based rating matrix, the matrix is usually sparse since each user only rates a few products. When one user changes rating or rate more, the average rating of that user changes and the normalization needs to be redone, leading to the need to redo the rating matrix as well. An approach that Amazon proposes, is to calculate the similarity among items then we can predict the rating of a new item based on those already rated items of the same user. Usually, the number of items are smaller than the number of users. This makes similarity matrix smaller and easier for calculation. Pick out a list of top 5-10 most similar items to the current one, rated by the user. Take their average or weighted average to indicate its rating. This method is developed by Amazon and can be used when there are more users than items, since it is more stable and faster. It is not very suited for datasets with browsing or entertainment related items.

# Model based

First to reduce the large but sparse user item matrix by matrix factorization. Matrix factorization is to breaking down a large matrix into a product of smaller ones. A = X * Y. For example a matrix of m users * n items can be factorized into the product of user matrix m * p features and item matrix n * p features. P are called latent features which means they are hidden characteristics of the user and the items.

## SVD 

One of the popular algorithm to factorize matrix into meaningful components is the singular value decomposition algorithm. This method was introduced at length in a previous post on LSD/LSA (laten semantic analysis) for document processing. It is to factorized the utility matrix into a multiplication of user and item matrix: $$ A = U \Sigma V^T $$

where U and V are orthogonal matrices, $$ \Sigma $$ being the diagonal matrix containing the singular values of A. The singular values in $$ \Sigma $$ are sorted in decreasing order and the number of non zero singular values indicates the rank of matrix A.

## PCA
PCA also decomposes the matrix into smaller matrices (in this case with eigenvectors). PCA algorithm has been introduced in a previous post.

## NMF

Non negative matrix factorization is a technique that decomposes a non negative matrix into the product of two non negative matrices: $$ V \approx W * H $$ with the constraint that V,W,and H does not contain any negative elements. The factorization is performed using an iterative optimization algorithm that minimizes the difference between V and the product of W and H: $$ \mid\mid V - WH \mid\mid^2 = \sum (V - WH)^2 $$. After having the two smaller matrices, we can predict the missing values using matrix multiplication. 

# Hybrid

A hybrid model is one that makes use of both methods (memory based and model based). This is to leverage the strengths and compensate for weaknesses of both methods. For example, a hybrid model might use the model based method to predict a rating for an item, and then use a memory based method to generate prediction from similar users or items. So that they can compare or combine those predictions in some ways. The hybrid model clearly enjoys the patterns learned by the model based method, at the same time can keep the personalization by the memory method.

# Deep learning 

## Auto encoder




```python

```
