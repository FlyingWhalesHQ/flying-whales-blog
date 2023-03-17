---
layout: post
title:  "Naive Bayes"
date:   2023-03-06 4:14:54 +0700
categories: jekyll update
---

# TOC

- [Definition](#define)
- [Maximum likelihood](#maxili)
- [Gaussian naive Bayes](#gnb)
- [Multinomial naive Bayes](#mnb)
- [Bernoulli naive Bayes](#bnb)

# Introduction <a name="intro"></a>

Thinking about our usual problem: given a feature vector and an output vector, naive Bayes classifiers would help us to classify a new input instance into one of the classes in the output vector. Naive Bayes classification makes use of Bayes’ theorem and some simple assumptions. Here is the Bayes’ theorem:

$$ P(A \mid B) = \frac{P(B \mid A)}{P(A)P(B)} $$

In plain English, $$ P(A\mid B) $$ is the probability of A given B. $$ P(B \mid A) $$ is the probability of B given A. P(A), P(B) are the probabilities of A and B respectively. This theorem simply states that posterior probability of an event A given B ($$ P(A \mid B) $$ ) would equal the prior probability of that event (P(A)) multiply with the likelihood of that event ($$ P(B \mid A) $$ which can also be explained as the probability of the evidence given the hypothesis that event A is true) divided by the evidence (P(B)). The smart and useful thing about this theorem is that it incorporates prior knowledge for you into your prediction, which in some cases, makes very educated guesses. One of the applications of Bayesian inference in the real world as of today is to classify documents (mainly). Another is on inferring whose DNA using biology knowledge (which sparks fundamental questions on ethics and privacy).

Let’s come back to the model and speak about the other component of naive Bayes: the two naive assumptions. First, we assume that the input x are conditionally independent from each other and second, that each input feature contributes equally to y the class prediction. These assumptions, despite being simple, work quite well in reality. To proceed, let's make use of some math symbol manipulations. How do those above assumptions translate into math notation? We first need to formulate the problem using mathematical symbols: Let X be the input vector, y be the prediction using X, into K classes. When we want to get the probability of y given those X, we use Bayes’ theorem:

$$ P(y|X) = \frac{P(X|y) P(y)}{P(X)} $$

Utilizing the naive assumptions:
	
$$ P(y|x1, x2,..., xn) = \frac{P(x_{1}|y) P(x_{2}|y) ...P(x_{n}|y)}{P(x_{1}) P (x_{2})...P(x_{n})P(y)} $$

Since the denominator is constant, we remove it and use proportionality instead:

$$ P(y \mid x_{1}, x_{2},..., x_{n}) P(x_{1} \mid y) P(x_{2}\mid y) ...P(x_{n}\mid y) P(y)=P(y)\prod_{i=1}{n}P(x_{i}\mid y) $$

# Maximum likelihood <a name="maxili"></a>

Applying the usual maximum likelihood principle: assume the highest probability that makes all the training y happens (that would give the prediction for the most possible class, too):

$$ k = arg max_{k} P(y \mid X) $$

Applying the transformation in the previous section:

$$ k = arg max_{k} P(y) \prod_{i=1}^{n}P(x_{i} \mid y) $$

From here, the calculation of y obviously depends on the type of distributions of $$ P(x_{i} \mid y) $$. Different classifiers are different in the way they make different assumptions on that distribution type. There are three main types: Gaussian naive Bayes, multinomial naive Bayes, and Bernoulli naive Bayes. With a given dataset, the posterior probabilities can be calculated from the data and then used as labels for a typical supervised learning problem using naive Bayes algorithms.


# Gaussian naive Bayes <a name="gnb"></a>

When we assume the conditional probability to be normally distributed, we make use of the following characteristics of that distribution:

Mean $$ \mu = \frac{1}{n}\sum_{i=1}{n}x_{i} $$

Standard deviation $$ \sigma =\sqrt{\frac{1}{n-1}\sum_{i=1}{n}(x_{i}- \mu )^{2}} $$

Density function $$ p(x)=\frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $$

Example 1: Consider a small dataset with two features: precipitation (rain level) and holiday, output: traffic or not. The precipitation level is 86 96 80 65 70 80 90 75 for days of traffic jams and 85 90 70 95 91 for days of traffic ok. Using two different set of rain levels (for two different types of traffic), we can calculate the mean and standard deviation for these two classes of prediction. Then we use those two different mean and standard deviation for the calculation of probability. That said, the probability of traffic jam with rain level of 74 would be:

$$ P(rain = 74 | jam = yes) = \frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{(74-\mu)^{2}}{2\sigma^{2}}}= 0.0344 $$

The probability of traffic ok with rain level 74 would be (this time with different mean and standard deviation):

$$ P(rain = 74 | jam = no) = \frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{(74-\mu)^{2}}{2\sigma^{2}}}= 0.0187 $$

Note that we need to normalize those predictions before using it in production: calculate all possible probabilities, sum them up and then divide each for the total to get the real probability.


Example 2: A credit institution decides whether to lend a credit line to a person based on several factors: age, income, student status, credit rate. In this case, let scale the income so that it comes from range 0 to 10. We encode the categorical attribute "student-or-not" into 0 and 1 with 0 being negative. Same for "lend-or-not" target. Credit rate also comes from 0 to 10.


|ID|Age|Income|Student|Credit rate|Lend or not?|
|--|--|--|--|--|--|
|1 |23|8 |0 |7 |0 |
|2 |45|9 |0 |5 |1 |
|3 |60|5 |0 |6 |1 |
|…|


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


X = [[23, 8, 0,7],[45,9,0,5],[60,5,0,6],[34,2,1,9],[14,4,0,4],
    [22, 8, 0,7],[40,4,1,5],[65,5,0,6],[35,2,1,9],[4,4,0,4],
    [25, 4, 1,5],[45,9,0,5],[60,2,0,4],[34,1,1,9],[14,2,0,4],
    [19, 8, 1,8],[42,6,0,7],[61,5,0,6],[34,2,1,10],[14,4,1,4]]
y = [0,1,1,1,1,
     0,1,1,1,1,
    1,1,0,0,0,
    1,1,1,0,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


 # training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)

```

    Accuracy: 62.5



# Multinomial naive Bayes <a name="mnb"></a>

In this classifier, $$ p(x_{i}\mid y) $$ is the frequency of $$ x_{i} $$ in the class $$ y_{j} $$

$$ p(x_{i}|y) = \frac{N_{x}}{N_{y}} $$

Consider a use case for multinomial naive Bayes, we need to classify documents. Then $$ N_{x} $$ is the total of times a word appears in the document of type k. $$ N_{y} $$ is the total number of words in that type. Sometimes a completely new word appears in production, leading to division by 0 and halting the process completely. We would make use of a technique called Laplace smoothing.

<b>Laplace smoothing</b>

Despite its fancy name, the rationale and the tweak are rather simple: to prevent 0 appearing in the frequency table and obstruct computation, we add 1 to each and every value in the equation. The reason for such a name is that this technique has mathematical interpretation, can be applied in other places, and specifically it affects the logic of the system and its ability to make inference.

Example 3: Classify a question on Quora to be sincere or insincere? Insincere means hate-speech or not-real.

<ul>
    <li>Preprocess data:
        <ul>
            <li> Remove numbers and punctuations </li>
            <li> Remove stopwords </li>
            <li> Stemming and lemmatization </li>
        </ul>
    </li>
    <li>Training model:
        <ul>
            <li> Find probability for each word. Eliminate words with probability smaller than 0.0001</li>
            <li> Find conditional probability = probability of that word / total (in)sincere words
            </li>
        </ul>
    </li>
    <li> Predict: (with Laplace smoothing) if insincere_term / total > 0.5 -> insincere
        <ul>
            <li> Calculate accuracy</li>
        </ul>
    </li>
</ul>




```python
import numpy as np 
import pandas as pd
train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

# preprocess
from sklearn.model_selection import train_test_split
train, test = train_test_split(train, test_size=0.2)
word_count = {}
word_count_sincere = {}
word_count_insincere = {}
sincere  = 0
insincere = 0 

import re
import string
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
row_count = train.shape[0]
for row in range(0,row_count):
    insincere += train.iloc[row]['target']
    sincere += (1 - train.iloc[row]['target'])
    sentence = train.iloc[row]['question_text']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = stemmer.stem(word)

    for word in words_in_sentence:
        if train.iloc[row]['target'] == 0:   #Sincere Words
            if word in word_count_sincere.keys():
                word_count_sincere[word]+=1
            else:
                word_count_sincere[word] = 1
        elif train.iloc[row]['target'] == 1: #Insincere Words
            if word in word_count_insincere.keys():
                word_count_insincere[word]+=1
            else:
                word_count_insincere[word] = 1
        if word in word_count.keys():        #For all words. I use this to compute probability.
            word_count[word]+=1
        else:
            word_count[word]=1

# find proba for each word. eliminate < 0.0001

word_probability = {}
total_words = 0
for i in word_count:
    total_words += word_count[i]
for i in word_count:
    word_probability[i] = word_count[i] / total_words

print ('Total words ',len(word_probability))
print ('Minimum probability ',min (word_probability.values()))
threshold_p = 0.0001
for i in list(word_probability):
    if word_probability[i] < threshold_p:
        del word_probability[i]
        if i in list(word_count_sincere):   #list(dict) return it;s key elements
            del word_count_sincere[i]
        if i in list(word_count_insincere):  
            del word_count_insincere[i]
print ('Total words ',len(word_probability))

# find conditional proba

total_sincere_words = sum(word_count_sincere.values())
cp_sincere = {}  #Conditional Probability
for i in list(word_count_sincere):
    cp_sincere[i] = word_count_sincere[i] / total_sincere_words

total_insincere_words = sum(word_count_insincere.values())
cp_insincere = {}  #Conditional Probability
for i in list(word_count_insincere):
    cp_insincere[i] = word_count_insincere[i] / total_insincere_words
    
#    predict

row_count = test.shape[0]
p_insincere = insincere / (sincere + insincere)
p_sincere = sincere / (sincere + insincere)
accuracy = 0

for row in range(0,row_count):
    sentence = test.iloc[row]['question_text']
    target = test.iloc[row]['target']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = stemmer.stem(word)

    insincere_term = p_insincere
    sincere_term = p_sincere
    
    sincere_M = len(cp_sincere.keys())
    insincere_M = len(cp_insincere.keys())
    for word in words_in_sentence:
        if word not in cp_insincere.keys():
            insincere_M +=1
        if word not in cp_sincere.keys():
            sincere_M += 1
         
    for word in words_in_sentence:
        if word in cp_insincere.keys():
            insincere_term *= (cp_insincere[word] + (1/insincere_M))
        else:
            insincere_term *= (1/insincere_M)
        if word in cp_sincere.keys():
            sincere_term *= (cp_sincere[word] + (1/sincere_M))
        else:
            sincere_term *= (1/sincere_M)
        
    if insincere_term/(insincere_term + sincere_term) > 0.5:
        response = 1
    else:
        response = 0
    if target == response:
        accuracy += 1
    
print ('Accuracy is ',accuracy/row_count*100)

# Accuracy is  94.13
```



# Bernoulli naive Bayes <a name="bnb"></a>

In this case, we only need to care about whether the word appears (i.e. we don’t care about its frequency). 

$$ p(x_{i} \mid y) = p(i \mid y)x_{i} (1-p(i \mid y))^{1-x_{i}} $$

In the above equation, if $$ x_{i} = 1 $$, its probability $$ p(x_{i} \mid y) =  p(i \mid y) $$ which is the probability the word appears in a document of class k. If $$ x_{i}=0 $$, its probability $$ p(x_{i} \mid y) =  1 - p(i \mid y) $$.

The likelihood of the document would then be:

$$ p(X \mid y) = \prod_{i=1}{n}p^{x_{i}}_{i}(1-p_{i})^{1-x_{i}} $$
