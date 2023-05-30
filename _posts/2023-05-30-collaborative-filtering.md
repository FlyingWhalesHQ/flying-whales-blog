---
layout: post
title:  "Collaborative Filtering"
date:   2023-05-30 10:14:54 +0700
categories: MachineLearning
---

# Intro

Collaborative filtering is a technique that is mainly used in recommendation system to suggest user's preferences by collecting data from a similar group of users. It assumes that if those people have similar preference and decisions in the past, so will they in the near future. Use case: Amazon develops item-to-item collaborative filtering in their recommendation system.

# Memory based

The memory based technique is to find users similar to the current user in the database and calculate the new rating based in the ratings of similar users found.

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
