---
layout: post
title:  "Interpretable AI: SHAP"
date:   2023-06-06 10:14:54 +0700
categories: MachineLearning
---
# TOC
- [Introduction](#)
- [Shapley value](#)
- [SHAP](#)
- [Code example](#)
    - [Titanic dataset](#)
    - [Home credit default risk dataset](#)
    - [Explaining VGG neural net](#)
- [Conclusion](#)

# Introduction

Continue our journey on exploring interpretability of complex models in AI, today we will look at another method to analyze machine learning model named SHAP. SHAP, short for Shapley Additive Explanations, aims to unify approaches before it, such as LIME, DeepLIFT, QII, Layer-wise relevance propagation, and the like. SHAP has a solid theoretical foundation and practical effectiveness since the indicator is based on Shapley value from cooperative game theory. It helps to explain the output of any machine learning model by attributing to each feature the change in the expected model prediction when we condition of that feature, given that these attributions are distributed fairly among the features. In other words, Shapley value is the marginal contribution of a feature into the model's final prediction. So it is an indicator of importance for that feature. In cooperative game theory, players can join into coalition and gets certain payoff from being in coalition. Applied into machine learning, each feature is a player and feature combinations are coalitions. For each feature in interest, we consider all the possible coalitions, then we calculate the contribution of that feature in each of those coalitions, by calculating the resulted difference in output when keeping it in and then leaving it out of the model. Then we take the average of those contributions to have the Shapley value for that feature. The upsides of SHAP are its unified nature and fairness. It can be used for linear models and deep learning models. It avoid certain biases in other techniques.

# Shapley value

Let's consider a toy example, a model to predict house price based on the area, address, house age, and whether it has a park nearby. The predicted price for a house of 50m2 near the center, newly built and not having a park nearby can be 200,000, 50,000 different from the average prediction by the model. Now we can ask the question, how much each of those features contributed to the prediction? It could be that the area of 50m2 contributed to -10,000 in the 50,000 difference, the addrress contributed to 40,000 in the difference, the house age contributed 20,000 and the having park nearby feature contributed 0. First, to calculate the contribution of the area of the house, we need to list out all the possible coalitions: 

- No features
- Address
- Address + House age
- Address + Park-nearby
- House age + Park-nearby

For each of these combinations we compute the predicted apartment price with and without the area feature and take the difference. Then we take the weighted average of those marginal contributions.

For image processing tasks, a player can be a super pixel (i.e. a group of pixels that are close and similar to each other).

# SHAP

Here is the SHAP equation for additive feature attribution methods:

$$ g(z) = \phi_0 + \sum_{j=1}^M \phi_j z_j $$

where g is the approximation model, M is the maximum coalition size, $$ \phi_j \in R $$ is the feature attribution for feature j (i.e. the Shapley value). 

Addtive feature attribution methods like LIME, DeepLIFT above have very desirable properties: local accuracy, missingness, and consistency. The first property is local accuracy which requires the explanation model to at least match the output of the complex model for the neighbourhood of that point.

$$ f(x) = g(x') = \phi_0 + \sum_{i=1}^M \phi_i x_i' $$

The explanation model would match the original model when we sample around the point x. The second property is missingness. Missing features in the original input will have no attributed impact. In other words, those features won't receive any attribution from the final outcome. The third property is consistency, when some input's contribution increases or stays the same, that input's attribution will not decrease.

# Code example

We will consider two datasets: Titanic and the home credit default risk. This is to see SHAP in action in different context so that we can see how the method usually works and what it can highlight and give us better understanding of both the dataset and the model.


```python
# Importing necessary libraries
import pandas as pd
import numpy as np
# import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# TITANIC

titanic = pd.read_csv('titanic.csv')
titanic.head()

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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We have 418 observations
titanic.shape
```




    (418, 12)




```python
# And Cabin has 327 missing values, so we will drop Cabin
titanic['Cabin'].isnull().sum()
```




    327




```python
# We also drop Name and Ticket 
titanic = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic = titanic.dropna()

# Splitting the dataset
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the RandomForestClassifier
model = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, y_train)

# Calculating SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualization of the first prediction's explanation 
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:])

```

<img width="883" alt="Screenshot 2023-06-06 at 15 19 52" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/6d2061c7-b358-487a-81c3-522c2667c75b">

The SHAP plot shows a detailed analysis for the prediction of the second observation in the test dataset (id = 0). The predicted survival rate is 0.05, with the main driving factor is the gender of the person. Being a male push the rate of survival really hard.


```python
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
sv = explainer(X_test)
exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_test.values, 
                  feature_names=X_test.columns)
idx = 0
waterfall(exp[idx])
```

The waterfall plot gives a closer looks at those feature component. Note that the expected survival rate is 0.342, but for this datapoint the predicted rate is only 0.047. The graph shows that the gender of the person drives down 0.27 his survival rate.

<img width="832" alt="Screenshot 2023-06-06 at 15 27 22" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/d4b3926d-320e-47d4-a401-9c897dfc0c89">


```python
shap.summary_plot(shap_values[1], X_test)

```

We can plot the summary of all those feature contribution. For example, in the titanic dataset, the gender of the person makes a large difference on the survival rate. If you are a man (red color) your survived rate is lowered for about -0.3. If you are a woman (blue dots), your survived rate would be increased by around 0.5. Other features pull and push but contribute around 0 to the final outcome.

<img width="797" alt="Screenshot 2023-06-06 at 15 29 17" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/cca8a41e-4aeb-4a7f-ac0c-3b64adcdebfc">


```python
shap.force_plot(explainer.expected_value[1], shap_values[1][:100,:], X_train_new.iloc[:100,:])

```

The push and pull of those features are not the same across all passengers

<img width="888" alt="Screenshot 2023-06-06 at 15 32 43" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/ae1e9bcd-d89e-43be-8ad8-b9a052a28719">

For the home credit default risk, we preprocess the data, use 10,000 observations only and pick the top 10 features. In the Titanic dataset, we already have a look at how SHAP explain a prediction for the test set. Here we would see the explanation for the first datapoint in the training data set.

For the first datapoint in the training set, the client did not default. The model predict the client to default at 0.27, the main factors being the credit ratings from external source 3 (toward defaulting) and external source 2 (toward not defaulting). The external source 3 pushes harder than external source 2. 

<img width="827" alt="Screenshot 2023-06-06 at 16 11 32" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/ff91ccc2-35c3-42e6-abd8-86e2db0f7c30">

To explain another observation using the waterfall plot, the predicted default rate is 0.02 (very low) compared to the expected prediction of 0.082. External source 3 still pushes the hardest and toward defaulting. It seems that external source 3 is a prominent but pessimistic credit rating scheme. Meanwhile, external source 2, number of employment days and other features predict toward not defaulting.

<img width="867" alt="Screenshot 2023-06-06 at 16 11 41" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/b13703f3-c4de-4ba7-a21a-b23e120b4d2d">

In the following picture you can see the summary of SHAP. It turns out that high values in external source 2's and external source 3's rating lower the default rate and very low ratings provided by these two sources can increase the default rate about 0.1-0.2. Other features push and pull at around 0.

<img width="849" alt="Screenshot 2023-06-06 at 16 12 00" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/d0133cdf-a052-4a51-b3f7-06a8b6faea57">

As above example, the effect of those features are not the same across observations.

<img width="850" alt="Screenshot 2023-06-06 at 16 12 21" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/1f27d647-43cf-4d02-becf-2b0681dfed83">

SHAP can be used for analyzing deep neural network, too. Here is an example that SHAP analyze the image's pixel and plot what contributes to the final prediction of VGG16 - a popular neural network used for image processing, pretrained on the ImageNet dataset. As usual, SHAP colors the contributing pixels toward the prediction of "golden retriever" in red and the against pixels in blue. From the resulting image that SHAP plots for us, we can see that VGG has identified the breed of the dog by mostly focusing on the heads of the animals in the picture. Which validates the model by proving that it has actually learned to see the dogs by looking at the dog, not to predict randomly based on some heuristic environment or trivial cues in the images. As previously mentioned, a black box might as well looks at the environment cues such as snow to predict the image to be an image of wolf instead of husky. But we can trust the VGG in this case, we can trust that it is making decision based on good reasoning and logic.


```python
# VGG NET

import numpy as np
import shap
import keras.applications.vgg16 as vgg16
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# Load VGG16 model
model = vgg16.VGG16(weights='imagenet')

# Load an image
img_path = 'cat-dog.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Initialize SHAP
explainer = shap.GradientExplainer(model, preprocess_input(np.zeros((1,224,224,3))))

# Compute SHAP values
shap_values,indexes = explainer.shap_values(x, ranked_outputs=1)

# Get class names
index_names = np.vectorize(lambda x: decode_predictions(np.eye(1, 1000, x))[0][0][1])(indexes)

# Plot SHAP values
shap.image_plot(shap_values, x, index_names)
```

<img width="489" alt="Screenshot 2023-06-06 at 16 50 59" src="https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/a4d8d4f2-7c2e-43ea-919b-d8efbda0c15a">


# Conclusion 

Since calculating SHAP exactly is challenging, SHAP comes with variants. There are model agnostic approximation methods, like Shapley smapling values and kernel SHAP to approximate Shapley values. There are model specific approximation methods, like Linear SHAP, Deep SHAP.

In conclusion, the need to balance accuracy and interpretability has motivated development of methods that aid in explaining predictions of machine learning models. This is a progressive step toward a better AI environment for end users, the non expert sets of people who also need the service of AI models. SHAP provides such explanation, built on sounding game theoretical concepts. It is able to provide consistent, locally accurate, and model agnostic explanations for individual predictions. The post has demonstrated how to use SHAP for both tabular and image data. It aids model understanding, feature importance, feature selection and model debugging when necessary. It can drill down to which words, which pixels or which features pushing or pulling the target, proving it to be an invaluable tool for a great range of users, from data analysts to business stakeholders.

SHAP contributes to setting new standards on machine learning models. That we don't just need them to be accurate quantitatively, we need them to be logically right also. We should be reassured that the model is looking at the right features when suggesting decisions for high stake cases. Having SHAP in the toolbox increase our trust in AI and AI assisted decision making processes.

