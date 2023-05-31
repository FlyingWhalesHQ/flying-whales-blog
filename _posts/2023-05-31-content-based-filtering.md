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

# Introduction

Content based filtering is a technique used in recommendation system that is based on the content (representing by features) of the items.

Let's start with the utility matrix: the columns being the users and the rows being the items. Since each user rates only a few items, the matrix is sparse, i.e. there are many missing values. If we can fill in those missing values, we can roughly know which items the user like and maybe suggest them those items. If we can classify the items into groups, when new item comes in the classification algorithm would predict the class the item belongs to and can also provide recommendation for the users accordingly.

For the rating, it can be explicit rating in which the user rates the items according to their preference or it can be implicit rating in which the user's preference is extrapolated based on the number of times they rewatch the videos, the amount of time they visit the product, or when the user actually buys that item.

# Content based recommendation

To build a content based recommendation system, we need to build a profile for each items. A profile is represented by a feature vector. For example, the features of a movie can be the actor, the director, the year and the genre.

# Code examples

Using the MovieLens dataset, we are going to do three examples: recommend movies based on genres, tags and ratings. When we recommend based on genres, it is the content of the movies that we care about. So if we want a movie for children, it is safer to use this method since it only recommends movies in proximity of the children genre. When we recommend based on tags, it is the comment of the users that we care about. And it is not just one user's comment, we aggregate the comments from different users on the same movies and generate new recommendation based on that. Doing this makes use of the preference of the user base. Surely some user might care about what other people say about the product and base their decision on that. When we recommend based on ratings, it is similar, this is for users who care about the opinion of the crowd to pick their next movie to watch. At the end, we can aggregate those three models and have a long list of recommendations minus the duplicates.

The techniques used are called TF-IDF and CountVectorizer. TF-IDF has been introduced in LSD/LSA articles. CountVectorizer in Python simply counts the number of appearance of the word in the documents, those are indicators of similarity.


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
def get_recommendations(title, movies, cosine_sim):
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

print(get_recommendations('Toy Story (1995)', movies, cosine_sim))

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
```


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

print(get_recommendations('Toy Story (1995)', movies_with_tags, cosine_sim))

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


```


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




```python
indices = pd.Series(movies_with_ratings.index, index=movies_with_ratings['title']).drop_duplicates()

print(get_recommendations('Toy Story (1995)', movies_with_ratings, cosine_sim))

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


In conclusion, content based filtering is a very popular method in building recommendation system. And we have reasonably good results. There are some flaws, though. When we ask to recommend by tags, the algorithm recommends "Pulp Fiction" for "Toy Story". This is a bit too much if we are talking about children customer. In the algorithm to recommend by ratings, it is normal to have a wide range of genres recommended and this is what the user might need. They might just need a 5 star movie, doesn't matter the topics. But for the algorithm that recommends based on users' tags, Toy Story and Pulp Fiction are still widely inappropriate in the case of children, even though it is not in the top 5 recommendations. The algorithm might just reveals some bias in the preference of the user, and it is not pretty.

In the next example, we get access to the genome scores of the tags of the movie lists. The genome score measures how relevant a tag is to a movie, and there is a tag list of a thousand. These tag scores have been calculated in advance by a machine learning algorithm, based on tags, ratings, and textual reviews. This genome score table provides a better overview of the movie's content for the machine to grasp. Then we can calculate the cosine similarities of those tag score vectors and use them for recommendation. The result is much better than before, since asking for recommendations on Toy Story returns very similar movies in the genres for children: Shrek, Ice Age, Monster Inc, Finding Nemo, Rattatouile, Up.


```python
genome_tags = pd.read_csv('ml-latest-small/genome_tags.csv')
genome_tags.head(15)
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
      <th>tagId</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>007 (series)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18th century</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1920s</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1930s</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1950s</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1960s</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1970s</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1980s</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>19th century</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>3d</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>70mm</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>80s</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>9/11</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>aardman</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Load the genome scores
genome_scores = pd.read_csv('ml-latest-small/genome_scores.csv')

# Pivot the genome scores DataFrame to create a movie-tag matrix
movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance')

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(movie_tag_matrix)

# Function that takes in movie title as input and outputs most similar movies
def get_recommended_movieIds(movieId, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the movieId
    idx = movie_tag_matrix.index.get_loc(movieId)

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movie_tag_matrix.index[movie_indices]
recommended = get_recommended_movieIds(1)
movies[movies['movieId'].isin(recommended)]
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
      <th>1706</th>
      <td>2294</td>
      <td>Antz (1998)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1757</th>
      <td>2355</td>
      <td>Bug's Life, A (1998)</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>3114</td>
      <td>Toy Story 2 (1999)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>4306</td>
      <td>Shrek (2001)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy|Ro...</td>
    </tr>
    <tr>
      <th>3568</th>
      <td>4886</td>
      <td>Monsters, Inc. (2001)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3745</th>
      <td>5218</td>
      <td>Ice Age (2002)</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>4360</th>
      <td>6377</td>
      <td>Finding Nemo (2003)</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>6405</th>
      <td>50872</td>
      <td>Ratatouille (2007)</td>
      <td>Animation|Children|Drama</td>
    </tr>
    <tr>
      <th>7039</th>
      <td>68954</td>
      <td>Up (2009)</td>
      <td>Adventure|Animation|Children|Drama</td>
    </tr>
    <tr>
      <th>7355</th>
      <td>78499</td>
      <td>Toy Story 3 (2010)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy|IMAX</td>
    </tr>
  </tbody>
</table>
</div>




If we have access to lengthy overview or critics of the movies, or if the items are documents, we can leverage NLP techniques to process and analyze the documents then calculate the cosine similarity. This algorithm can be used to suggest title for blog posts, or suggest new movies to watch.

Let's try to use three movie plots. NLP's preprocessing task involves lowercasing, removing punctuations, removing stopwords, lemmatizing etc. Here we would skip this step. We would use the Word2Vec model (which comes with its latent word space) to train further on this new corpus of three plots. The number of dimensions for each word is 100. We consider a window of 10 words for each word. Word occurring less than five times would be ignored. Then we process the three plots. Each plot would have their own number of words. Each of these words are represented as a Word2Vec vector (100 dimensional vector). Each vector would then be averaged. So that we have a resulting list of three plots, each consists of 100 numerically represented words. Then we calculate the cosine similarity among those plots.

From the result, we can see that these three movies are not similar with similarity indices from 0.1 to 0.2. All those movies are in 2020: Demon Slayer (Japanese and anime based), the Eight Hundred (Chinese war drama), Bad Boys for Life (American action comedy). We can see that these movies come from very different cultures.


```python
from gensim.models import Word2Vec

with open('movies/movie1.txt', 'r') as f:
    m1 = f.read()
with open('movies/movie2.txt', 'r') as f:
    m2 = f.read()
with open('movies/movie3.txt', 'r') as f:
    m3 = f.read()

```


```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text (convert text into list of words)
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize words (convert words into their root form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Example usage:
preprocessed_text1 = preprocess_text(m1)
preprocessed_text2 = preprocess_text(m2)
preprocessed_text3 = preprocess_text(m3)

print(preprocessed_text2)

```

    ['early', 'day', 'second', 'sinojapanese', 'war', 'greater', 'scale', 'world', 'war', 'ii', 'imperial', 'japanese', 'army', 'invaded', 'shanghai', 'became', 'known', 'battle', 'shanghai', 'holding', 'back', 'japanese', '3', 'month', 'suffering', 'heavy', 'loss', 'chinese', 'army', 'forced', 'retreat', 'due', 'danger', 'encircled', 'lieutenant', 'colonel', 'xie', 'jinyuan', '524th', 'regiment', 'underequipped', '88th', 'division', 'national', 'revolutionary', 'army', 'led', '452', 'young', 'officer', 'soldier', 'defend', 'sihang', 'warehouse', '3rd', 'imperial', 'japanese', 'division', 'consisting', 'around', '20000', 'troop', 'heroic', 'suicidal', 'last', 'stand', 'japanese', 'order', 'generalissimo', 'nationalist', 'china', 'chiang', 'kaishek', 'decision', 'made', 'provide', 'morale', 'boost', 'chinese', 'people', 'loss', 'beijing', 'shanghai', 'helped', 'spur', 'support', 'western', 'power', 'full', 'view', 'battle', 'international', 'settlement', 'shanghai', 'across', 'suzhou', 'creek', '6']


    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/nguyenlinhchi/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/nguyenlinhchi/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/nguyenlinhchi/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to
    [nltk_data]     /Users/nguyenlinhchi/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!



```python
documents = [preprocessed_text1,preprocessed_text2,preprocessed_text3]

# Assume that `documents` is a list of strings, where each string is a movie's plot summary or reviews
# documents = [doc.split() for doc in documents]  # split each document into words

# Train a Word2Vec model
model = Word2Vec(documents, size=100, window=5, min_count=1, workers=4)

# Vectorize the movies
movie_vectors = [np.mean([model.wv[word] for word in doc], axis=0) for doc in documents]

# Compute the similarity matrix
similarity_matrix = cosine_similarity(movie_vectors)

# Now you can use this similarity matrix to recommend similar movies

```


```python
similarity_matrix
```




    array([[0.99999994, 0.11791225, 0.17876801],
           [0.11791225, 1.0000001 , 0.12657738],
           [0.17876801, 0.12657738, 0.99999994]], dtype=float32)




```python

```
