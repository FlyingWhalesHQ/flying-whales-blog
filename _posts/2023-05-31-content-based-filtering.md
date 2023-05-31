---
layout: post
title:  "Content-based Filtering"
date:   2023-05-31 10:14:54 +0700
categories: MachineLearning
---

# TOC
- [Introduction](#intro)
- [Content based recommendation](#content)
- [Code examples](#code)
- [Conclusion](#conclude)

# Introduction

Content based filtering is a technique used in recommendation system that is based on the content of items, i.e. the features (that can be hidden).

Let's start with the utility matrix: with the columns being the users and the rows being the items. Since each user rates only a few items, the matrix is sparse, i.e. there are many missing values. If we can fill in those missing values, we can roughly know which items the user like and maybe suggest them those items. If we can classify the items into groups, when new item comes in the classification algorithm would predict the class the item belongs to and can also provide recommendation for the users accordingly.

For the rating, it can be explicit rating in which the user rates the items according to their preference or it can be implicit rating in which the user's preference is extrapolated based on the number of times they rewatch the videos, the amount of time they visit the product, or when the user actually buys that item.

# Content based recommendation

To build a content based recommendation system, we need to profile all the items. A profile is represented by a feature vector. For example, the features of a song can be the singer, the author, the year and the genre.

# Code examples

Using the MovieLens dataset, we are going to do three examples: recommend movies based on genres, tags and ratings. When we recommend based on genres, it is the content of the movies that we care about. So if we want a movie for children, it is safer to use this method since it only recommends movies in proximity of the children genre. When we recommend based on tags, it is the comment of the users that we care about. And it is not just one user's comment, we aggregate the comments from different users on the same movies and generate new recommendation based on that. Doing this makes use of the preference of the user base. Surely some user might care about what other people say about the product and base their decision on that. When we recommend based on ratings, it is similar, this is for users who care about the opinion of the crowd to pick their next movie to watch. At the end, we can aggregate those three models and have a long list of recommendations minus the duplicates.


```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Movies and genre
movies = pd.read_csv('ml-latest-small/movies.csv', low_memory=False)

len(movies)
```




    9742




```python
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define a TF-IDF Vectorizer Object.
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
movies['genres'] = movies['genres'].fillna('')

# Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

```


```python
#Construct a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

print(get_recommendations('Toy Story (1995)'))

```

    1706                                          Antz (1998)
    2355                                   Toy Story 2 (1999)
    2809       Adventures of Rocky and Bullwinkle, The (2000)
    3000                     Emperor's New Groove, The (2000)
    3568                                Monsters, Inc. (2001)
    6194                                     Wild, The (2006)
    6486                               Shrek the Third (2007)
    6948                       Tale of Despereaux, The (2008)
    7760    Asterix and the Vikings (Astérix et les Viking...
    8219                                         Turbo (2013)
    Name: title, dtype: object


Apart from the genres, we also have the information about the movie's tags by user. This kind of information reveals a bit more about the preference of users on a movie.


```python
# Load the tags data
tags = pd.read_csv('ml-latest-small/tags.csv')
tags.head()
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
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>60756</td>
      <td>funny</td>
      <td>1445714994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>60756</td>
      <td>Highly quotable</td>
      <td>1445714996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>60756</td>
      <td>will ferrell</td>
      <td>1445714992</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>89774</td>
      <td>Boxing story</td>
      <td>1445715207</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>89774</td>
      <td>MMA</td>
      <td>1445715200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge movies and tags into a single DataFrame
movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')
movies_with_tags.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336.0</td>
      <td>pixar</td>
      <td>1.139046e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>474.0</td>
      <td>pixar</td>
      <td>1.137207e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>567.0</td>
      <td>fun</td>
      <td>1.525286e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62.0</td>
      <td>fantasy</td>
      <td>1.528844e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62.0</td>
      <td>magic board game</td>
      <td>1.528844e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(movies_with_tags)
```




    11853




```python
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

# Concatenate all tags for each movie into a single string
movies_with_tags['tags'] = movies_with_tags.groupby('movieId')['tag'].transform(lambda x: ' '.join(x))

# Replace NaN with an empty string
movies_with_tags['tags'] = movies_with_tags['tags'].fillna('')

# Remove duplicate movies
movies_with_tags = movies_with_tags.drop_duplicates(subset=["movieId"])


# Define a TF-IDF Vectorizer Object
tfidf = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by applying the fit_transform method on the tags feature
tfidf_matrix = tfidf.fit_transform(movies_with_tags['tags'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

```


```python
movies_with_tags.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>timestamp</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336.0</td>
      <td>pixar</td>
      <td>1.139046e+09</td>
      <td>pixar pixar fun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62.0</td>
      <td>fantasy</td>
      <td>1.528844e+09</td>
      <td>fantasy magic board game Robin Williams game</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>289.0</td>
      <td>moldy</td>
      <td>1.143425e+09</td>
      <td>moldy old</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>474.0</td>
      <td>pregnancy</td>
      <td>1.137374e+09</td>
      <td>pregnancy remake</td>
    </tr>
  </tbody>
</table>
</div>




```python
cosine_sim
```




    array([[1., 0., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
#Construct a reverse map of indices and movie titles
indices = pd.Series(movies_with_tags.index, index=movies_with_tags['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_with_tags['title'].iloc[movie_indices]

# Get recommendations for a specific movie, for example, "The Dark Knight"
print(get_recommendations('Toy Story (1995)'))

```

    2484                 Bug's Life, A (1998)
    3210                   Toy Story 2 (1999)
    10675    Guardians of the Galaxy 2 (2017)
    8664                            Up (2009)
    10485                   Big Hero 6 (2014)
    10240               The Lego Movie (2014)
    9459                 Avengers, The (2012)
    395                   Pulp Fiction (1994)
    3                          Jumanji (1995)
    7                 Grumpier Old Men (1995)
    Name: title, dtype: object



```python
from sklearn.feature_extraction.text import CountVectorizer

# Load ratings data
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Calculate the average rating for each movie
average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.rating = average_ratings.rating.round()

# Merge the average rating to the movie data
movies_with_ratings = movies.merge(average_ratings, on='movieId')

# Define a CountVectorizer Object to create a matrix where each row will represent a movie and each column will represent a different user's rating
count = CountVectorizer()

# Construct the required matrix by applying the fit_transform method on the title feature and average rating feature
count_matrix = count.fit_transform(movies_with_ratings['title'] + ' ' + movies_with_ratings['rating'].astype(str))

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(movies_with_ratings.index, index=movies_with_ratings['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_with_ratings['title'].iloc[movie_indices]
print(get_recommendations('Toy Story (1995)'))

```

    2353                   Toy Story 2 (1999)
    7338                   Toy Story 3 (2010)
    256     Pyromaniac's Love Story, A (1995)
    1                          Jumanji (1995)
    5                             Heat (1995)
    6                          Sabrina (1995)
    9                        GoldenEye (1995)
    12                           Balto (1995)
    13                           Nixon (1995)
    15                          Casino (1995)
    Name: title, dtype: object



```python
movies_with_ratings.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>3.920930</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>3.431818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>3.259615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>2.357143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>3.071429</td>
    </tr>
  </tbody>
</table>
</div>




```python
cosine_sim
```




    array([[1.        , 0.28867513, 0.2236068 , ..., 0.        , 0.        ,
            0.        ],
           [0.28867513, 1.        , 0.25819889, ..., 0.        , 0.        ,
            0.        ],
           [0.2236068 , 0.25819889, 1.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            1.        ]])



# Conclusion

In conclusion, content based filtering is a very popular method in building recommendation system. And we have reasonably good results. There are some flaws, though. When we ask to recommend by tags, the algorithm recommends "Pulp Fiction" for "Toy Story". This is a bit too much if we are talking about children customer. In the algorithm to recommend by ratings, it is normal to have a wide range of genres recommended and this is what the user might need. They might just need a 5 star movie, doesn't matter the topics. But for the algorithm that recommends based on users' tags, Toy Story and Pulp Fiction are still widely inappropriate in the case of children, even though it is not in the top 5 recommendations. The algorithm might just reveals some bias in the preference of the user, and it is not pretty.
