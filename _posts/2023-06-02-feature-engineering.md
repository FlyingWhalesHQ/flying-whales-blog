---
layout: post
title:  "Feature Engineering"
date:   2023-06-02 10:14:54 +0700
categories: MachineLearning
---

# TOC
- [Introduction](#)
- [Code example](#)
    - [Preprocessing](#)
    - [Building living-condition feature](#)
    - [Adding financial-stress feature](#)
    - [Adding previous loan situation](#)
- [Conclusion](#)

# Introduction
Intuitively, feature engineering is the process of understanding the data intimately. So that we can handcraft new features that represent the dataset better and improve the prediction of the model.

Some common methods are:

- Binning/bucketing: For example, in a dataset about the home credit default rate, when collecting client's data of their age, it could make better sense to divide the range into categories: less than 20 years old, from 20 to 30 years old, from 30 to 40 years old, from 40 to 50 years old, and above 50 years old. The reasoning behind this division is that, client less than 20 years old are not allowed to take a loan, and client above 50 years old can be groupped into one group since the most popular ages to take loans are from 20 to 50. Then we diving equally from the age 20 to 50. This unequal division of ages into buckets actually make better sense and generalize the age groups better.

- Polynomial features: We can take square of features, for example, to assume that those features having a nonlinear relationship with the target.

- Log transform the variables with long tailed distribution so that the new logged feature has a normal distribution

- Feature interaction: This is a way to combine different features, by assuming them having relationship among themselves. For example, we can combine family related features of a client together (which can be a simple linear combination or a complicated equation). The new feature would represent an overview of the client's family status. 

- Categorical feature handling: Since we usually need to transform categorical feature into numerical one, there are ways to do it such as onehot encoding (encode the value into a vector of 1 and 0s, with 1 being the cateogry it belongs to) or label encoding (encode each category as a different number).

- Date time variables: If we have the data on date and time, we can add a lagged variable (the value of the feature in some day in the past), calculate the interval between two dates (for example, the age of the house/car of the client who comes to request a loan).

- Scale the feature: since features are different in nature, they naturally use different units and scales. But that would makes the model inaccurate since the model doesn't really grasp the differences in scales. We can do some engineering to bring all features into one scale, in a way, for the machine to understand the dataset a bit better. The most two popular ways is to do minmax scaling and standardization. In min max scaling, we scale each feature back to a range, could be from 0 to 1. This is also called normalization. In standardization, we minus each value to the mean and divided by the standard deviation of the sample.

# Code example

The things noted above are general advice. In reality, the feature engineering process depends on the nature of the dataset itself (its dimensions, its purpose, the underlying patterns). Today we explore the the dataset for the home credit default risk. When we look into the features, we can see that there are about 50 features about the building that the client lives in. We can combine those features into a new one named "living_condition" by machine learning technique such as kernal PCA and Kmeans algorithm. Then we can add a financial_stress variable by considering a weighted combination of common factors such as credit income ration, current income, number of children, family size, spouse situation and other bills. Thirdly, add an statistical aggregation of previous loan application to add credibility and credit worthiness of the client into the dataset. Finally, we can consider some other features such as red flag that takes into account credit evaluation from external sources and then non linear relationship with the target.

## Preprocessing


```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(suppress=True)

```


```python
labels = pd.read_csv('home-credit-default-risk/HomeCredit_columns_description.csv',encoding='ISO-8859-1')
data = pd.read_csv('home-credit-default-risk/application_train.csv')
```


```python
labels.head()
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
      <th>Unnamed: 0</th>
      <th>Table</th>
      <th>Row</th>
      <th>Description</th>
      <th>Special</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>application_{train|test}.csv</td>
      <td>SK_ID_CURR</td>
      <td>ID of loan in our sample</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>application_{train|test}.csv</td>
      <td>TARGET</td>
      <td>Target variable (1 - client with payment diffi...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>application_{train|test}.csv</td>
      <td>NAME_CONTRACT_TYPE</td>
      <td>Identification if loan is cash or revolving</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>application_{train|test}.csv</td>
      <td>CODE_GENDER</td>
      <td>Gender of the client</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>application_{train|test}.csv</td>
      <td>FLAG_OWN_CAR</td>
      <td>Flag if the client owns a car</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# First take all the name of the features related to the building
living_condition = labels['Row'][44:91]
living_condition
```




    44                  APARTMENTS_AVG
    45                BASEMENTAREA_AVG
    46     YEARS_BEGINEXPLUATATION_AVG
    47                 YEARS_BUILD_AVG
    48                  COMMONAREA_AVG
    49                   ELEVATORS_AVG
    50                   ENTRANCES_AVG
    51                   FLOORSMAX_AVG
    52                   FLOORSMIN_AVG
    53                    LANDAREA_AVG
    54            LIVINGAPARTMENTS_AVG
    55                  LIVINGAREA_AVG
    56         NONLIVINGAPARTMENTS_AVG
    57               NONLIVINGAREA_AVG
    58                 APARTMENTS_MODE
    59               BASEMENTAREA_MODE
    60    YEARS_BEGINEXPLUATATION_MODE
    61                YEARS_BUILD_MODE
    62                 COMMONAREA_MODE
    63                  ELEVATORS_MODE
    64                  ENTRANCES_MODE
    65                  FLOORSMAX_MODE
    66                  FLOORSMIN_MODE
    67                   LANDAREA_MODE
    68           LIVINGAPARTMENTS_MODE
    69                 LIVINGAREA_MODE
    70        NONLIVINGAPARTMENTS_MODE
    71              NONLIVINGAREA_MODE
    72                 APARTMENTS_MEDI
    73               BASEMENTAREA_MEDI
    74    YEARS_BEGINEXPLUATATION_MEDI
    75                YEARS_BUILD_MEDI
    76                 COMMONAREA_MEDI
    77                  ELEVATORS_MEDI
    78                  ENTRANCES_MEDI
    79                  FLOORSMAX_MEDI
    80                  FLOORSMIN_MEDI
    81                   LANDAREA_MEDI
    82           LIVINGAPARTMENTS_MEDI
    83                 LIVINGAREA_MEDI
    84        NONLIVINGAPARTMENTS_MEDI
    85              NONLIVINGAREA_MEDI
    86              FONDKAPREMONT_MODE
    87                  HOUSETYPE_MODE
    88                  TOTALAREA_MODE
    89              WALLSMATERIAL_MODE
    90             EMERGENCYSTATE_MODE
    Name: Row, dtype: object




```python
# Now preprocess the data a bit
data.head()
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>




```python
y_train = data['TARGET']
X_train = data.drop(['TARGET'], axis=1)
y_train = y_train.to_frame()
y_train
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
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>1</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 1 columns</p>
</div>




```python
# Let's handle categorical / numerical variables and missing values

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categoricals = ['object']

X_train_categorical = X_train.select_dtypes(include=categoricals)
X_train_numerical = X_train.select_dtypes(include=numerics)

categorical_columns = X_train_categorical.columns
numerical_columns = X_train_numerical.columns

categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# imputer = imputer.fit(X_train)
  
X_train_categorical = categorical_imputer.fit_transform(X_train_categorical)
X_train_categorical = pd.DataFrame(data=X_train_categorical, columns=categorical_columns)

X_train_numerical = numerical_imputer.fit_transform(X_train_numerical)
X_train_numerical = pd.DataFrame(data=X_train_numerical, columns=numerical_columns)

```

The thing about using label encoder instead of one hot encoder is that in label encoder, there is an inherent assumption that the values are hierarchically meaningful. This might or might not reflect the qualitative meaning of the value in reality. For example, we categorize the house into 3 district: district 1, district 2, district 3 and encode them into number 0, 1, and 2. Since 2 > 1, it might suggest that district 2 is better than district 1 which might not reflect the real situation in which there are no inherent difference in those two geographical locations (they are both equal in distance to the center for example). We might take this inherent bias into account and try to make a new variable (via clustering or via distance to center) to compensate for this bias in the model. The same goes for the days of the week, inherently the meaning of monday tuesday to sunday might not be that linear. We can hope that the model might have enough data to learn this representation. One hot encoding, on the other hand, assume those categories are all equal, and it puts 1 for that category and 0s for others in the representation vector. For example: a house in district 1 can be represented as [0,1,0].

### Building living_condition feature


```python
X_train_categorical = X_train_categorical.apply(LabelEncoder().fit_transform)
X_train_categorical
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
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>OCCUPATION_TYPE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>ORGANIZATION_TYPE</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>37</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>307506</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>14</td>
      <td>4</td>
      <td>43</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>57</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>4</td>
      <td>39</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 16 columns</p>
</div>




```python
# Some of the features are categorical ('FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE')
# the rest is numerical
living_condition_categoricals = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
# living_condition_numericals = [e in living_condition if e not in living_condition_categoricals]
living_condition_numericals = np.setdiff1d(living_condition,living_condition_categoricals)
X_train_numerical[living_condition_numericals]
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
      <th>APARTMENTS_AVG</th>
      <th>APARTMENTS_MEDI</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_AVG</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>BASEMENTAREA_MODE</th>
      <th>COMMONAREA_AVG</th>
      <th>COMMONAREA_MEDI</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_AVG</th>
      <th>...</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_AVG</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>YEARS_BUILD_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.02470</td>
      <td>0.02500</td>
      <td>0.025200</td>
      <td>0.036900</td>
      <td>0.036900</td>
      <td>0.038300</td>
      <td>0.014300</td>
      <td>0.014400</td>
      <td>0.014400</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014900</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.619200</td>
      <td>0.624300</td>
      <td>0.634100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.09590</td>
      <td>0.09680</td>
      <td>0.092400</td>
      <td>0.052900</td>
      <td>0.052900</td>
      <td>0.053800</td>
      <td>0.060500</td>
      <td>0.060800</td>
      <td>0.049700</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.009800</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.071400</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.796000</td>
      <td>0.798700</td>
      <td>0.804000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.028358</td>
      <td>0.028236</td>
      <td>0.027022</td>
      <td>0.102547</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.028358</td>
      <td>0.028236</td>
      <td>0.027022</td>
      <td>0.102547</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.028358</td>
      <td>0.028236</td>
      <td>0.027022</td>
      <td>0.102547</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
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
      <th>307506</th>
      <td>0.20210</td>
      <td>0.20400</td>
      <td>0.100800</td>
      <td>0.088700</td>
      <td>0.088700</td>
      <td>0.017200</td>
      <td>0.020200</td>
      <td>0.020300</td>
      <td>0.017200</td>
      <td>0.220000</td>
      <td>...</td>
      <td>0.109500</td>
      <td>0.111800</td>
      <td>0.012500</td>
      <td>0.289800</td>
      <td>0.987600</td>
      <td>0.987600</td>
      <td>0.978200</td>
      <td>0.830000</td>
      <td>0.832300</td>
      <td>0.712500</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0.02470</td>
      <td>0.02500</td>
      <td>0.025200</td>
      <td>0.043500</td>
      <td>0.043500</td>
      <td>0.045100</td>
      <td>0.002200</td>
      <td>0.002200</td>
      <td>0.002200</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021400</td>
      <td>0.972700</td>
      <td>0.972700</td>
      <td>0.972700</td>
      <td>0.626000</td>
      <td>0.631000</td>
      <td>0.640600</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0.10310</td>
      <td>0.10410</td>
      <td>0.105000</td>
      <td>0.086200</td>
      <td>0.086200</td>
      <td>0.089400</td>
      <td>0.012300</td>
      <td>0.012400</td>
      <td>0.012400</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.797000</td>
      <td>0.981600</td>
      <td>0.981600</td>
      <td>0.981600</td>
      <td>0.748400</td>
      <td>0.751800</td>
      <td>0.758300</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>0.01240</td>
      <td>0.01250</td>
      <td>0.012600</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.028358</td>
      <td>0.028236</td>
      <td>0.027022</td>
      <td>0.008600</td>
      <td>0.977100</td>
      <td>0.977100</td>
      <td>0.977200</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0.07420</td>
      <td>0.07490</td>
      <td>0.075600</td>
      <td>0.052600</td>
      <td>0.052600</td>
      <td>0.054600</td>
      <td>0.017600</td>
      <td>0.017700</td>
      <td>0.017800</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071800</td>
      <td>0.988100</td>
      <td>0.988100</td>
      <td>0.988100</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 43 columns</p>
</div>




```python
X_train_categorical[living_condition_categoricals]
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
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 4 columns</p>
</div>




```python
X_train_living_condition = pd.concat([X_train_numerical[living_condition_numericals], X_train_categorical[living_condition_categoricals]],axis=1)
X_train_living_condition
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
      <th>APARTMENTS_AVG</th>
      <th>APARTMENTS_MEDI</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_AVG</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>BASEMENTAREA_MODE</th>
      <th>COMMONAREA_AVG</th>
      <th>COMMONAREA_MEDI</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_AVG</th>
      <th>...</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_AVG</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>YEARS_BUILD_MODE</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.02470</td>
      <td>0.02500</td>
      <td>0.025200</td>
      <td>0.036900</td>
      <td>0.036900</td>
      <td>0.038300</td>
      <td>0.014300</td>
      <td>0.014400</td>
      <td>0.014400</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.619200</td>
      <td>0.624300</td>
      <td>0.634100</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.09590</td>
      <td>0.09680</td>
      <td>0.092400</td>
      <td>0.052900</td>
      <td>0.052900</td>
      <td>0.053800</td>
      <td>0.060500</td>
      <td>0.060800</td>
      <td>0.049700</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.796000</td>
      <td>0.798700</td>
      <td>0.804000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
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
      <th>307506</th>
      <td>0.20210</td>
      <td>0.20400</td>
      <td>0.100800</td>
      <td>0.088700</td>
      <td>0.088700</td>
      <td>0.017200</td>
      <td>0.020200</td>
      <td>0.020300</td>
      <td>0.017200</td>
      <td>0.220000</td>
      <td>...</td>
      <td>0.987600</td>
      <td>0.987600</td>
      <td>0.978200</td>
      <td>0.830000</td>
      <td>0.832300</td>
      <td>0.712500</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0.02470</td>
      <td>0.02500</td>
      <td>0.025200</td>
      <td>0.043500</td>
      <td>0.043500</td>
      <td>0.045100</td>
      <td>0.002200</td>
      <td>0.002200</td>
      <td>0.002200</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.972700</td>
      <td>0.972700</td>
      <td>0.972700</td>
      <td>0.626000</td>
      <td>0.631000</td>
      <td>0.640600</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0.10310</td>
      <td>0.10410</td>
      <td>0.105000</td>
      <td>0.086200</td>
      <td>0.086200</td>
      <td>0.089400</td>
      <td>0.012300</td>
      <td>0.012400</td>
      <td>0.012400</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.981600</td>
      <td>0.981600</td>
      <td>0.981600</td>
      <td>0.748400</td>
      <td>0.751800</td>
      <td>0.758300</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>0.01240</td>
      <td>0.01250</td>
      <td>0.012600</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977100</td>
      <td>0.977100</td>
      <td>0.977200</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0.07420</td>
      <td>0.07490</td>
      <td>0.075600</td>
      <td>0.052600</td>
      <td>0.052600</td>
      <td>0.054600</td>
      <td>0.017600</td>
      <td>0.017700</td>
      <td>0.017800</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.988100</td>
      <td>0.988100</td>
      <td>0.988100</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 47 columns</p>
</div>



Since the dataset is quite large, I use only the first 10000 observations. Then we transform the X_train_living_condition into higher dimensional space with the RBF kernel (Radial basis function) where they can be separated better. After that, we use K-means to determine the clusters. The ELBOW shows that 3 clusters is optimal. Which roughly means that the living condition of clients can be groupped into three clusters. We will create a new feature with those new cluster labels.


```python
X_train_living_condition = X_train_living_condition[:10000]
```


```python
X_train_living_condition
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
      <th>APARTMENTS_AVG</th>
      <th>APARTMENTS_MEDI</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_AVG</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>BASEMENTAREA_MODE</th>
      <th>COMMONAREA_AVG</th>
      <th>COMMONAREA_MEDI</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_AVG</th>
      <th>...</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_AVG</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>YEARS_BUILD_MODE</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.02470</td>
      <td>0.02500</td>
      <td>0.025200</td>
      <td>0.036900</td>
      <td>0.036900</td>
      <td>0.038300</td>
      <td>0.014300</td>
      <td>0.014400</td>
      <td>0.014400</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.972200</td>
      <td>0.619200</td>
      <td>0.624300</td>
      <td>0.634100</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.09590</td>
      <td>0.09680</td>
      <td>0.092400</td>
      <td>0.052900</td>
      <td>0.052900</td>
      <td>0.053800</td>
      <td>0.060500</td>
      <td>0.060800</td>
      <td>0.049700</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.985100</td>
      <td>0.796000</td>
      <td>0.798700</td>
      <td>0.804000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
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
      <th>9995</th>
      <td>0.01630</td>
      <td>0.01670</td>
      <td>0.016800</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.080000</td>
      <td>...</td>
      <td>0.980600</td>
      <td>0.980600</td>
      <td>0.980600</td>
      <td>0.632800</td>
      <td>0.637700</td>
      <td>0.647200</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0.11744</td>
      <td>0.11785</td>
      <td>0.114231</td>
      <td>0.088442</td>
      <td>0.087955</td>
      <td>0.087543</td>
      <td>0.044621</td>
      <td>0.044595</td>
      <td>0.042553</td>
      <td>0.078942</td>
      <td>...</td>
      <td>0.977735</td>
      <td>0.977752</td>
      <td>0.977065</td>
      <td>0.752471</td>
      <td>0.755746</td>
      <td>0.759637</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 47 columns</p>
</div>




```python
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# assume df is your DataFrame
X = X_train_living_condition.values

# Kernel PCA transformation using RBF
kpca = KernelPCA(kernel="rbf")
X_kpca = kpca.fit_transform(X)

# finding the optimal number of clusters for KMeans after transformation
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_kpca)
    kmeanModel.fit(X_kpca)
    distortions.append(sum(np.min(cdist(X_kpca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_kpca.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Apply KMeans with the optimal number of clusters (you can find it from the above plot)
kmeans = KMeans(n_clusters=3)  # here 3 is used as an example, replace it with your optimal number
kmeans.fit(X_kpca)

# getting the cluster labels for each sample
labels = kmeans.labels_
```

![Elbow for optimal number of clusters in living ocndition](https://github.com/FlyingWhalesHQ/flying-whales-blog/assets/7457301/52842bc2-e79a-4bde-80c4-e97f3ef79bfb)


```python
X_train_living_condition['living_condition_cluster_label'] = labels
the_rest = []
for e in X_train.columns:
    if e not in living_condition:
        the_rest.append(e)
the_rest
X_train_the_rest = X_train[the_rest][:10000]
X_train_new = pd.concat([X_train_the_rest, X_train_living_condition], axis=1)

```

Now we can create more features based on this new way of clustering living condition. For example, we can take the mean, min, max and sum of the income of those living condition brackets. They are different in the sum of income but the mean is similar. 

|living_condition_cluster_label|bincount|income mean|min|max|sum|
|--|--|--|--|--|--|
|0|7300|164439|25650|1935000|1000000000|
|1|454|188565|36000|810000|80000000|
|2|2246|172962|33300|810000|300000000|

We can see that the group (2) with smallest size (454/10000) having the highest mean income (180000), even though summing up, they are the least (80000000). The group (0) with biggest size (7300) having the lowest mean income (160000), even though summing up, they are the highest. The biggest group also has the biggest range of income, ranging from 25000 to 2000000. These are the internal structure of the dataset that the kernelPCA and the Kmeans discover. Then we can merge this living situation feature into the original one.


```python
np.bincount(labels)
```


```python
X_train_living_situation = X_train_new.groupby('living_condition_cluster_label').agg({'AMT_INCOME_TOTAL': ['mean', 'min', 'max', 'sum'],
                                                        # Add more columns as needed
                                                        }).reset_index()
X_train_living_situation
X_train_living_situation.columns = ['living_condition_cluster_label', 'AMT_INCOME_TOTAL_mean','AMT_INCOME_TOTAL_min','AMT_INCOME_TOTAL_max','AMT_INCOME_TOTAL_sum' ] 
X_train_new = X_train_new.merge(X_train_living_situation, on='living_condition_cluster_label', how='left')

```


```python
X_train_new
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
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>...</th>
      <th>AMT_CREDIT_min</th>
      <th>AMT_CREDIT_sum</th>
      <th>AMT_ANNUITY_mean</th>
      <th>AMT_ANNUITY_max</th>
      <th>AMT_ANNUITY_min</th>
      <th>DAYS_DECISION_mean</th>
      <th>DAYS_DECISION_max</th>
      <th>DAYS_DECISION_min</th>
      <th>CNT_PAYMENT_mean</th>
      <th>CNT_PAYMENT_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>...</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775000</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>-606.000000</td>
      <td>-606.0</td>
      <td>-606.0</td>
      <td>24.000000</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>...</td>
      <td>68053.5</td>
      <td>1452573.0</td>
      <td>56553.990000</td>
      <td>98356.995</td>
      <td>6737.310</td>
      <td>-1305.000000</td>
      <td>-746.0</td>
      <td>-2341.0</td>
      <td>10.000000</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>...</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250000</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>-815.000000</td>
      <td>-815.0</td>
      <td>-815.0</td>
      <td>4.000000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2625259.5</td>
      <td>23651.175000</td>
      <td>39954.510</td>
      <td>2482.920</td>
      <td>-272.444444</td>
      <td>-181.0</td>
      <td>-617.0</td>
      <td>23.000000</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>...</td>
      <td>14616.0</td>
      <td>999832.5</td>
      <td>12278.805000</td>
      <td>22678.785</td>
      <td>1834.290</td>
      <td>-1222.833333</td>
      <td>-374.0</td>
      <td>-2357.0</td>
      <td>20.666667</td>
      <td>124.0</td>
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
      <th>307506</th>
      <td>456251</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>254700.0</td>
      <td>27558.0</td>
      <td>225000.0</td>
      <td>...</td>
      <td>40455.0</td>
      <td>40455.0</td>
      <td>6605.910000</td>
      <td>6605.910</td>
      <td>6605.910</td>
      <td>-273.000000</td>
      <td>-273.0</td>
      <td>-273.0</td>
      <td>8.000000</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>456252</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>72000.0</td>
      <td>269550.0</td>
      <td>12001.5</td>
      <td>225000.0</td>
      <td>...</td>
      <td>56821.5</td>
      <td>56821.5</td>
      <td>10074.465000</td>
      <td>10074.465</td>
      <td>10074.465</td>
      <td>-2497.000000</td>
      <td>-2497.0</td>
      <td>-2497.0</td>
      <td>6.000000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>456253</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>153000.0</td>
      <td>677664.0</td>
      <td>29979.0</td>
      <td>585000.0</td>
      <td>...</td>
      <td>13945.5</td>
      <td>41251.5</td>
      <td>4770.405000</td>
      <td>5567.715</td>
      <td>3973.095</td>
      <td>-2380.000000</td>
      <td>-1909.0</td>
      <td>-2851.0</td>
      <td>5.000000</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>456254</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>171000.0</td>
      <td>370107.0</td>
      <td>20205.0</td>
      <td>319500.0</td>
      <td>...</td>
      <td>21456.0</td>
      <td>268879.5</td>
      <td>10681.132500</td>
      <td>19065.825</td>
      <td>2296.440</td>
      <td>-299.500000</td>
      <td>-277.0</td>
      <td>-322.0</td>
      <td>15.000000</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>456255</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>675000.0</td>
      <td>49117.5</td>
      <td>675000.0</td>
      <td>...</td>
      <td>45000.0</td>
      <td>3395448.0</td>
      <td>20775.391875</td>
      <td>54022.140</td>
      <td>2250.000</td>
      <td>-587.625000</td>
      <td>-171.0</td>
      <td>-991.0</td>
      <td>21.750000</td>
      <td>174.0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 133 columns</p>
</div>



### Adding financial_stress feature
There are multiple factors for the financial stress, but some main ones are: credit-income ratio, current income, current credit line, the number of children, the number of family members, spouse's income, spouse's credit line, bills. We can weight those factors differently, too, since they affect the client differently. 



```python
original = pd.concat([X_train[['AMT_INCOME_TOTAL','AMT_CREDIT','CNT_FAM_MEMBERS','CNT_CHILDREN']],X_train_categorical['NAME_FAMILY_STATUS']],axis=1)
original
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
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>CNT_CHILDREN</th>
      <th>NAME_FAMILY_STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>157500.0</td>
      <td>254700.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>72000.0</td>
      <td>269550.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>153000.0</td>
      <td>677664.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>171000.0</td>
      <td>370107.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>157500.0</td>
      <td>675000.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 5 columns</p>
</div>




```python
def compute_financial_stress(row):
    w_credit_income_ratio = 2
    w_family_size = 1
    w_family_status = 1 
    w_children = 1.5
    stress_score = (row['AMT_CREDIT'] / row['AMT_INCOME_TOTAL']) * w_credit_income_ratio + row['CNT_FAM_MEMBERS'] * w_family_size + row['NAME_FAMILY_STATUS'] * w_family_status + row['CNT_CHILDREN'] * w_children
    return stress_score

original['financial_stress'] = original.apply(compute_financial_stress, axis=1)
original
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
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>CNT_CHILDREN</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>financial_stress</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>8.015778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>12.581500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>6.632333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>12.444444</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>157500.0</td>
      <td>254700.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>6.234286</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>72000.0</td>
      <td>269550.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
      <td>13.487500</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>153000.0</td>
      <td>677664.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>11.858353</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>171000.0</td>
      <td>370107.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>7.328737</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>157500.0</td>
      <td>675000.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>11.571429</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 6 columns</p>
</div>




### Add previous application situation
Previous application situation is an agregation of previous loans by the same person. It might rougly tell the credibility of the person, plus other important information.


```python
previous_application = pd.read_csv('home-credit-default-risk/previous_application.csv')
previous_application.head()
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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
# Aggregate based on the SK_ID_CURR
prev_app_agg = previous_application.groupby('SK_ID_CURR').agg({'AMT_CREDIT': ['mean', 'max', 'min', 'sum'],
                                                    'AMT_ANNUITY': ['mean', 'max', 'min'],
                                                    'DAYS_DECISION': ['mean', 'max', 'min'],
                                                    'CNT_PAYMENT': ['mean', 'sum']})

# Handle multi-level column names
prev_app_agg.columns = ['_'.join(col).strip() for col in prev_app_agg.columns.values]

# Reset the index
prev_app_agg.reset_index(inplace=True)
prev_app_agg
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
      <th>SK_ID_CURR</th>
      <th>AMT_CREDIT_mean</th>
      <th>AMT_CREDIT_max</th>
      <th>AMT_CREDIT_min</th>
      <th>AMT_CREDIT_sum</th>
      <th>AMT_ANNUITY_mean</th>
      <th>AMT_ANNUITY_max</th>
      <th>AMT_ANNUITY_min</th>
      <th>DAYS_DECISION_mean</th>
      <th>DAYS_DECISION_max</th>
      <th>DAYS_DECISION_min</th>
      <th>CNT_PAYMENT_mean</th>
      <th>CNT_PAYMENT_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>23787.00</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>-1740.000</td>
      <td>-1740</td>
      <td>-1740</td>
      <td>8.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775000</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>-606.000</td>
      <td>-606</td>
      <td>-606</td>
      <td>24.00</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>68053.5</td>
      <td>1452573.0</td>
      <td>56553.990000</td>
      <td>98356.995</td>
      <td>6737.310</td>
      <td>-1305.000</td>
      <td>-746</td>
      <td>-2341</td>
      <td>10.00</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250000</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>-815.000</td>
      <td>-815</td>
      <td>-815</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>20076.75</td>
      <td>40153.5</td>
      <td>0.0</td>
      <td>40153.5</td>
      <td>4813.200000</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>-536.000</td>
      <td>-315</td>
      <td>-757</td>
      <td>12.00</td>
      <td>12.0</td>
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
    </tr>
    <tr>
      <th>338852</th>
      <td>456251</td>
      <td>40455.00</td>
      <td>40455.0</td>
      <td>40455.0</td>
      <td>40455.0</td>
      <td>6605.910000</td>
      <td>6605.910</td>
      <td>6605.910</td>
      <td>-273.000</td>
      <td>-273</td>
      <td>-273</td>
      <td>8.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>338853</th>
      <td>456252</td>
      <td>56821.50</td>
      <td>56821.5</td>
      <td>56821.5</td>
      <td>56821.5</td>
      <td>10074.465000</td>
      <td>10074.465</td>
      <td>10074.465</td>
      <td>-2497.000</td>
      <td>-2497</td>
      <td>-2497</td>
      <td>6.00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>338854</th>
      <td>456253</td>
      <td>20625.75</td>
      <td>27306.0</td>
      <td>13945.5</td>
      <td>41251.5</td>
      <td>4770.405000</td>
      <td>5567.715</td>
      <td>3973.095</td>
      <td>-2380.000</td>
      <td>-1909</td>
      <td>-2851</td>
      <td>5.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>338855</th>
      <td>456254</td>
      <td>134439.75</td>
      <td>247423.5</td>
      <td>21456.0</td>
      <td>268879.5</td>
      <td>10681.132500</td>
      <td>19065.825</td>
      <td>2296.440</td>
      <td>-299.500</td>
      <td>-277</td>
      <td>-322</td>
      <td>15.00</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>338856</th>
      <td>456255</td>
      <td>424431.00</td>
      <td>1271929.5</td>
      <td>45000.0</td>
      <td>3395448.0</td>
      <td>20775.391875</td>
      <td>54022.140</td>
      <td>2250.000</td>
      <td>-587.625</td>
      <td>-171</td>
      <td>-991</td>
      <td>21.75</td>
      <td>174.0</td>
    </tr>
  </tbody>
</table>
<p>338857 rows × 13 columns</p>
</div>




```python
X_train_new = X_train.merge(prev_app_agg, on='SK_ID_CURR', how='left')
X_train_new
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
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>...</th>
      <th>AMT_CREDIT_min</th>
      <th>AMT_CREDIT_sum</th>
      <th>AMT_ANNUITY_mean</th>
      <th>AMT_ANNUITY_max</th>
      <th>AMT_ANNUITY_min</th>
      <th>DAYS_DECISION_mean</th>
      <th>DAYS_DECISION_max</th>
      <th>DAYS_DECISION_min</th>
      <th>CNT_PAYMENT_mean</th>
      <th>CNT_PAYMENT_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>...</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775000</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>-606.000000</td>
      <td>-606.0</td>
      <td>-606.0</td>
      <td>24.000000</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>...</td>
      <td>68053.5</td>
      <td>1452573.0</td>
      <td>56553.990000</td>
      <td>98356.995</td>
      <td>6737.310</td>
      <td>-1305.000000</td>
      <td>-746.0</td>
      <td>-2341.0</td>
      <td>10.000000</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>...</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250000</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>-815.000000</td>
      <td>-815.0</td>
      <td>-815.0</td>
      <td>4.000000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2625259.5</td>
      <td>23651.175000</td>
      <td>39954.510</td>
      <td>2482.920</td>
      <td>-272.444444</td>
      <td>-181.0</td>
      <td>-617.0</td>
      <td>23.000000</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>...</td>
      <td>14616.0</td>
      <td>999832.5</td>
      <td>12278.805000</td>
      <td>22678.785</td>
      <td>1834.290</td>
      <td>-1222.833333</td>
      <td>-374.0</td>
      <td>-2357.0</td>
      <td>20.666667</td>
      <td>124.0</td>
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
      <th>307506</th>
      <td>456251</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>254700.0</td>
      <td>27558.0</td>
      <td>225000.0</td>
      <td>...</td>
      <td>40455.0</td>
      <td>40455.0</td>
      <td>6605.910000</td>
      <td>6605.910</td>
      <td>6605.910</td>
      <td>-273.000000</td>
      <td>-273.0</td>
      <td>-273.0</td>
      <td>8.000000</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>456252</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>72000.0</td>
      <td>269550.0</td>
      <td>12001.5</td>
      <td>225000.0</td>
      <td>...</td>
      <td>56821.5</td>
      <td>56821.5</td>
      <td>10074.465000</td>
      <td>10074.465</td>
      <td>10074.465</td>
      <td>-2497.000000</td>
      <td>-2497.0</td>
      <td>-2497.0</td>
      <td>6.000000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>456253</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>153000.0</td>
      <td>677664.0</td>
      <td>29979.0</td>
      <td>585000.0</td>
      <td>...</td>
      <td>13945.5</td>
      <td>41251.5</td>
      <td>4770.405000</td>
      <td>5567.715</td>
      <td>3973.095</td>
      <td>-2380.000000</td>
      <td>-1909.0</td>
      <td>-2851.0</td>
      <td>5.000000</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>456254</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>171000.0</td>
      <td>370107.0</td>
      <td>20205.0</td>
      <td>319500.0</td>
      <td>...</td>
      <td>21456.0</td>
      <td>268879.5</td>
      <td>10681.132500</td>
      <td>19065.825</td>
      <td>2296.440</td>
      <td>-299.500000</td>
      <td>-277.0</td>
      <td>-322.0</td>
      <td>15.000000</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>456255</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>675000.0</td>
      <td>49117.5</td>
      <td>675000.0</td>
      <td>...</td>
      <td>45000.0</td>
      <td>3395448.0</td>
      <td>20775.391875</td>
      <td>54022.140</td>
      <td>2250.000</td>
      <td>-587.625000</td>
      <td>-171.0</td>
      <td>-991.0</td>
      <td>21.750000</td>
      <td>174.0</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 133 columns</p>
</div>



### Misc
A feature of red-flag can be added based on credit rate from external sources. As before, the external sources can have different credibility. And then some polynomial features can be added to show non linear relationship. Here we will only demonstrate how to apply polynomial conversion to one feature: the goods price that the client applies for. Since it is not clear that the price of the goods would behave linearly with the default risk, we apply quadratic transformation.


```python
original = X_train_numerical[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
original
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
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>0.510853</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.502130</td>
      <td>0.555912</td>
      <td>0.729567</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502130</td>
      <td>0.650442</td>
      <td>0.510853</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.502130</td>
      <td>0.322738</td>
      <td>0.510853</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>0.145570</td>
      <td>0.681632</td>
      <td>0.510853</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0.502130</td>
      <td>0.115992</td>
      <td>0.510853</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0.744026</td>
      <td>0.535722</td>
      <td>0.218859</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>0.502130</td>
      <td>0.514163</td>
      <td>0.661024</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0.734460</td>
      <td>0.708569</td>
      <td>0.113922</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 3 columns</p>
</div>




```python
def compute_red_flag(row):
    w_source_1 = 2
    w_source_2 = 1
    w_source_3 = 1 
    red_flag = row['EXT_SOURCE_1'] * w_source_1 + row['EXT_SOURCE_2'] * w_source_2 + row['EXT_SOURCE_3'] * w_source_3
    return red_flag

original['red_flag'] = original.apply(compute_red_flag, axis=1)
original
```

    /var/folders/kf/5_ggvsz93vxdbx_h0tvy66xh0000gn/T/ipykernel_15715/1336237161.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      original['red_flag'] = original.apply(compute_red_flag, axis=1)





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
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>red_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
      <td>0.568398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>0.510853</td>
      <td>1.755633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.502130</td>
      <td>0.555912</td>
      <td>0.729567</td>
      <td>2.289738</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502130</td>
      <td>0.650442</td>
      <td>0.510853</td>
      <td>2.165554</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.502130</td>
      <td>0.322738</td>
      <td>0.510853</td>
      <td>1.837851</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307506</th>
      <td>0.145570</td>
      <td>0.681632</td>
      <td>0.510853</td>
      <td>1.483626</td>
    </tr>
    <tr>
      <th>307507</th>
      <td>0.502130</td>
      <td>0.115992</td>
      <td>0.510853</td>
      <td>1.631105</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>0.744026</td>
      <td>0.535722</td>
      <td>0.218859</td>
      <td>2.242634</td>
    </tr>
    <tr>
      <th>307509</th>
      <td>0.502130</td>
      <td>0.514163</td>
      <td>0.661024</td>
      <td>2.179446</td>
    </tr>
    <tr>
      <th>307510</th>
      <td>0.734460</td>
      <td>0.708569</td>
      <td>0.113922</td>
      <td>2.291411</td>
    </tr>
  </tbody>
</table>
<p>307511 rows × 4 columns</p>
</div>




```python
# Quadratic transformation
original = X_train_numerical['AMT_GOODS_PRICE'][:10000]

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, interaction_only=True)
poly.fit_transform([original])
original
```




    0        351000.0
    1       1129500.0
    2        135000.0
    3        297000.0
    4        513000.0
              ...    
    9995     270000.0
    9996     900000.0
    9997     450000.0
    9998     315000.0
    9999     270000.0
    Name: AMT_GOODS_PRICE, Length: 10000, dtype: float64



# Conclusion
So far we have studied some forms of feature engineering in the wild. And the examples use non trivial methods to engineer the features. Those examples reflect the domain knowledge necessary in due diligence of financial institution when they consider credit lines/cash for consumers and other knowledge as a data scientist when it comes to process large dataset. To recap, we have seen that there are many input features regarding the building in which the client lives in, these are indirect reflect of their living condition, hence we create a new variable based on those input features. Then we see that there can be a financial stress evaluation based on the situation of the client. That indicator would be helpful since we are wondering whether the client can pay back the loan or not. For example, the credit-income ratio can tell how much the credit line would weigh the person's income down, especially if they have children. We can see that the dataset lacks the information on the spouse situation (their income and credit), so one logical thing to do is that we might come back to the field and collect such information, to make our prediction more sensible. The third feature we engineer is the situation of the previous loan applications of the same client. The past can say a lot about the present and the future of this same client. So that we can aggregate those statistics into the current model's calculation.

Together, those examples provide an overview of how to do feature engineering for a dataset, which is a crucial process and it can affect the model's performance directly. Translating into real world business, it can help the institution makes better decision in aiding people in need.
