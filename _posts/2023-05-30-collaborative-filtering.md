---
layout: post
title:  "Collaborative Filtering"
date:   2023-05-30 10:14:54 +0700
categories: MachineLearning
---

# Intro

Collaborative filtering is a technique that is mainly used in recommendation system to suggest user's preferences by collecting data from a similar group of users. It assumes that if those people have similar preference and decisions in the past, so will they in the near future. Use case: Amazon develops item-to-item collaborative filtering in their recommendation system.

# Memory based

The memory based technique is to find users or items similar to the current user in the database and calculate the new rating based in the ratings of similar users or items found.

Let's start with a user-item rating matrix. We have the colums to be users and rows to be items. For large companies such as Netflix, Amazon, etc, the matrix can be quite large with thousands of users and items. One issue with this is that they might not be able to obtain all ratings to fill in this matrix since each user might rate a few items only. So the matrix is quite sparse. There are several options to fill in those missing values. First is to place 0 for all the missing values. This is not a good choice since 0 means the lowest rating possible. A better option is to replace N/A with the average. For example, if the lowest is 0 and highest possible is 5, we can use 2.5. However, this has some issue with different kind of users. For example, an easy reviewer might rate 5 stars for a likeable move and for movies she doesn't like, she rate 3. If we replace her N/A with 2.5, all those movies mean bad movies for her. But for a difficult reviewer, she might only give 3 stars for movies she really likes and 1 star for movies she doesn't like. If we replace all her N/A with 2.5, we are assuming that she likes all the rest. A more reasonale option is to use a different average which is the average of all the ratings she has done. This alleviates the problem of easy and difficult user mentioned above. After filling those numbers in, we can normalize the matrix by substrating each user's rating column by their own average. This effectively set all the missing values to 0. Then 0 would be the neutral value and a positive value would mean that that user likes that movie. Likewise, a negative value means that the user doesn't like that movie.

How to calculate similarity?

## Euclidean distance

## Cosine distance
Based on the angle.
## Centered cosine distance
Factor out the average ratings.

## Cosine similarity


Choose a list of top 5-10 most similar users, take average of their ratings. $$ R_U = \frac{\sum_{u=1}^n R_u}{n} $$
This is to take all the top similar users equally.
Or weigh them accordingly: $$ R_U = \frac{\sum_{u=1}^n R_u * W_u}{\sum_{u=1}^n W_u} $$. The obvious weights are the similarity index for each user. 

## User based

## Item based
Pick out a list of top 5-10 most similar items to the current one, rated by the user. Take their average or weighted average to indicate its rating.

This is developed by Amazon and can be used when there are more users than items, since it is more stable and faster. It is not very suited for datasets with browsing or entertainment related items.

# Model based

First to reduce the large but sparse user item matrix by matrix factorization. Matrix factorization is to breaking down a large matrix into a product of smaller ones. A = X * Y. For example a matrix of m users * n items can be factorized into the product of user matrix m * p features and item matrix n * p features. P are called latent features which means they are hidden characteristics of the user and the items.

## SVD 

One of the popular algorithm to factorize matrix into meaningful components is the singular value decomposition algorithm.

## PCA

## NMF

## Neural network (auto encoder)




```python

```
