---
layout: post
title:  "Feature Selection"
date:   2023-06-1 10:14:54 +0700
categories: MachineLearning
---

# Introduction

Feature selection is a technique in machine learning that only choose a subset of all the available features to construct a model. We know that not all features contributes to the prediction equally. For example, to predict a house price, the size of house can be much more important than the address of the house. So feature selection can help to reduce the number of variables, that would reduce computation cost and training time and make the model more parsimonious. The more important thing is that those variables are qualitatively different, so if there is an algorithm or procedure to select only the most contributing attributes, it would be better. This can combat overfitting and help with the ultimate goal of generalization, since the model sees the underlying pattern of the data. That would also possible make the model perform better.

Feature selection is a part of feature engineering (create new variables that make better sense). And it is part of the process of understanding your data qualitatively. So make sure you use logics and domain knowledge to evaluate each feature thoroughly before deciding to drop or to add features.

Automatically, there are ways to do feature selection: 

- Filter method: This is to score the relevance of each feature to the target variable, making use of statistical measures such as Chi-Squared test, information gain, correlation coefficients.
- Wrapper method: Why not use a model to predict the importance of the features? The search can be an exhaustive algorithm, or heuristics such as forward selection, backward elemination, recursive feature eliminiation, genetic algorithm.
- Embedded method: The model to select features is embeded into the final model construction process, using models such as LASSO, elastic net, decision tree, random forest. This method can be more efficient than the above.

# Filter method

## Chi squared test
This statistical test is to evaluate the likelihood of correlation between variables using their distribution


```python
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

```


```python
train = pd.read_csv('home-credit-manual-engineered-features/train_previous_raw.csv', nrows = 1000)
test = pd.read_csv('home-credit-manual-engineered-features/test_previous_raw.csv', nrows = 1000)
```


```python
train.head()
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
      <th>client_installments_AMT_PAYMENT_min_sum</th>
      <th>client_installments_AMT_PAYMENT_sum_mean</th>
      <th>client_installments_AMT_PAYMENT_sum_max</th>
      <th>client_installments_AMT_PAYMENT_sum_min</th>
      <th>client_installments_AMT_PAYMENT_sum_sum</th>
      <th>client_installments_counts_mean</th>
      <th>client_installments_counts_max</th>
      <th>client_installments_counts_min</th>
      <th>client_installments_counts_sum</th>
      <th>client_counts</th>
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
      <td>175783.725</td>
      <td>219625.695000</td>
      <td>219625.695</td>
      <td>219625.695</td>
      <td>4.172888e+06</td>
      <td>19.000000</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>361.0</td>
      <td>19.0</td>
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
      <td>1154108.295</td>
      <td>453952.220400</td>
      <td>1150977.330</td>
      <td>80773.380</td>
      <td>1.134881e+07</td>
      <td>9.160000</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>229.0</td>
      <td>25.0</td>
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
      <td>16071.750</td>
      <td>21288.465000</td>
      <td>21288.465</td>
      <td>21288.465</td>
      <td>6.386539e+04</td>
      <td>3.000000</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>3.0</td>
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
      <td>994476.690</td>
      <td>232499.719688</td>
      <td>691786.890</td>
      <td>25091.325</td>
      <td>3.719996e+06</td>
      <td>7.875000</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>16.0</td>
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
      <td>483756.390</td>
      <td>172669.901591</td>
      <td>280199.700</td>
      <td>18330.390</td>
      <td>1.139621e+07</td>
      <td>13.606061</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>898.0</td>
      <td>66.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1166 columns</p>
</div>




```python
y_train = train['TARGET']
train = train.drop(['TARGET'], axis=1)
```


```python
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
      <th>995</th>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 1 columns</p>
</div>




```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X_train = train.select_dtypes(include=numerics)
columns = X_train.columns
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train)
  
X_train = imputer.transform(X_train)
X_train = pd.DataFrame(data=X_train, columns=columns)
X_train[X_train < 0] = 0

```


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
```


```python
chi2_selector = SelectKBest(chi2, k=800) # Select top 5 features
X_train_kbest = chi2_selector.fit_transform(X_train, y_train)

# To get the selected feature names
selected_features = [feature for feature, mask in zip(X_train.columns, chi2_selector.get_support()) if mask]
print(selected_features)

```

    ['SK_ID_CURR', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'COMMONAREA_AVG', 'LANDAREA_AVG', 'NONLIVINGAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI', 'LANDAREA_MEDI', 'NONLIVINGAREA_MEDI', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'previous_loans_NAME_CONTRACT_TYPE_Cash loans_count', 'previous_loans_NAME_CONTRACT_TYPE_Consumer loans_count', 'previous_loans_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_loans_NAME_CONTRACT_TYPE_XNA_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_FRIDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_FRIDAY_count_norm', 'previous_loans_WEEKDAY_APPR_PROCESS_START_MONDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_MONDAY_count_norm', 'previous_loans_WEEKDAY_APPR_PROCESS_START_SATURDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_SUNDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_SUNDAY_count_norm', 'previous_loans_WEEKDAY_APPR_PROCESS_START_THURSDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_THURSDAY_count_norm', 'previous_loans_WEEKDAY_APPR_PROCESS_START_TUESDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_TUESDAY_count_norm', 'previous_loans_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count', 'previous_loans_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count_norm', 'previous_loans_FLAG_LAST_APPL_PER_CONTRACT_N_count', 'previous_loans_FLAG_LAST_APPL_PER_CONTRACT_N_count_norm', 'previous_loans_FLAG_LAST_APPL_PER_CONTRACT_Y_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a new car_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a new car_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a used car_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Buying a used car_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Car repairs_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Car repairs_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Education_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Furniture_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Journey_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Medicine_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Medicine_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Other_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Repairs_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Repairs_count_norm', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Urgent needs_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_XAP_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_XNA_count', 'previous_loans_NAME_CASH_LOAN_PURPOSE_XNA_count_norm', 'previous_loans_NAME_CONTRACT_STATUS_Approved_count', 'previous_loans_NAME_CONTRACT_STATUS_Approved_count_norm', 'previous_loans_NAME_CONTRACT_STATUS_Canceled_count', 'previous_loans_NAME_CONTRACT_STATUS_Canceled_count_norm', 'previous_loans_NAME_CONTRACT_STATUS_Refused_count', 'previous_loans_NAME_CONTRACT_STATUS_Refused_count_norm', 'previous_loans_NAME_CONTRACT_STATUS_Unused offer_count', 'previous_loans_NAME_CONTRACT_STATUS_Unused offer_count_norm', 'previous_loans_NAME_PAYMENT_TYPE_Cash through the bank_count', 'previous_loans_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count', 'previous_loans_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count_norm', 'previous_loans_NAME_PAYMENT_TYPE_Non-cash from your account_count', 'previous_loans_NAME_PAYMENT_TYPE_Non-cash from your account_count_norm', 'previous_loans_NAME_PAYMENT_TYPE_XNA_count', 'previous_loans_CODE_REJECT_REASON_CLIENT_count', 'previous_loans_CODE_REJECT_REASON_CLIENT_count_norm', 'previous_loans_CODE_REJECT_REASON_HC_count', 'previous_loans_CODE_REJECT_REASON_HC_count_norm', 'previous_loans_CODE_REJECT_REASON_LIMIT_count_norm', 'previous_loans_CODE_REJECT_REASON_SCO_count', 'previous_loans_CODE_REJECT_REASON_SCO_count_norm', 'previous_loans_CODE_REJECT_REASON_SCOFR_count', 'previous_loans_CODE_REJECT_REASON_SCOFR_count_norm', 'previous_loans_CODE_REJECT_REASON_SYSTEM_count', 'previous_loans_CODE_REJECT_REASON_VERIF_count', 'previous_loans_CODE_REJECT_REASON_VERIF_count_norm', 'previous_loans_CODE_REJECT_REASON_XAP_count', 'previous_loans_CODE_REJECT_REASON_XAP_count_norm', 'previous_loans_CODE_REJECT_REASON_XNA_count', 'previous_loans_NAME_TYPE_SUITE_Children_count', 'previous_loans_NAME_TYPE_SUITE_Children_count_norm', 'previous_loans_NAME_TYPE_SUITE_Family_count', 'previous_loans_NAME_TYPE_SUITE_Family_count_norm', 'previous_loans_NAME_TYPE_SUITE_Group of people_count', 'previous_loans_NAME_TYPE_SUITE_Group of people_count_norm', 'previous_loans_NAME_TYPE_SUITE_Other_A_count', 'previous_loans_NAME_TYPE_SUITE_Other_A_count_norm', 'previous_loans_NAME_TYPE_SUITE_Other_B_count', 'previous_loans_NAME_TYPE_SUITE_Other_B_count_norm', 'previous_loans_NAME_TYPE_SUITE_Spouse, partner_count', 'previous_loans_NAME_TYPE_SUITE_Spouse, partner_count_norm', 'previous_loans_NAME_TYPE_SUITE_Unaccompanied_count', 'previous_loans_NAME_CLIENT_TYPE_New_count', 'previous_loans_NAME_CLIENT_TYPE_New_count_norm', 'previous_loans_NAME_CLIENT_TYPE_Refreshed_count', 'previous_loans_NAME_CLIENT_TYPE_Repeater_count', 'previous_loans_NAME_CLIENT_TYPE_Repeater_count_norm', 'previous_loans_NAME_CLIENT_TYPE_XNA_count', 'previous_loans_NAME_CLIENT_TYPE_XNA_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Audio/Video_count', 'previous_loans_NAME_GOODS_CATEGORY_Auto Accessories_count', 'previous_loans_NAME_GOODS_CATEGORY_Clothing and Accessories_count', 'previous_loans_NAME_GOODS_CATEGORY_Clothing and Accessories_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Computers_count', 'previous_loans_NAME_GOODS_CATEGORY_Computers_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Construction Materials_count', 'previous_loans_NAME_GOODS_CATEGORY_Construction Materials_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Consumer Electronics_count', 'previous_loans_NAME_GOODS_CATEGORY_Consumer Electronics_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Direct Sales_count', 'previous_loans_NAME_GOODS_CATEGORY_Furniture_count', 'previous_loans_NAME_GOODS_CATEGORY_Furniture_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Gardening_count', 'previous_loans_NAME_GOODS_CATEGORY_Homewares_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Jewelry_count', 'previous_loans_NAME_GOODS_CATEGORY_Jewelry_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Medical Supplies_count', 'previous_loans_NAME_GOODS_CATEGORY_Medical Supplies_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Medicine_count', 'previous_loans_NAME_GOODS_CATEGORY_Medicine_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Mobile_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Office Appliances_count', 'previous_loans_NAME_GOODS_CATEGORY_Office Appliances_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Other_count', 'previous_loans_NAME_GOODS_CATEGORY_Other_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_count', 'previous_loans_NAME_GOODS_CATEGORY_Sport and Leisure_count', 'previous_loans_NAME_GOODS_CATEGORY_Tourism_count', 'previous_loans_NAME_GOODS_CATEGORY_Tourism_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Vehicles_count', 'previous_loans_NAME_GOODS_CATEGORY_Vehicles_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_Weapon_count', 'previous_loans_NAME_GOODS_CATEGORY_Weapon_count_norm', 'previous_loans_NAME_GOODS_CATEGORY_XNA_count', 'previous_loans_NAME_GOODS_CATEGORY_XNA_count_norm', 'previous_loans_NAME_PORTFOLIO_Cards_count', 'previous_loans_NAME_PORTFOLIO_Cards_count_norm', 'previous_loans_NAME_PORTFOLIO_Cars_count', 'previous_loans_NAME_PORTFOLIO_Cars_count_norm', 'previous_loans_NAME_PORTFOLIO_Cash_count', 'previous_loans_NAME_PORTFOLIO_POS_count', 'previous_loans_NAME_PORTFOLIO_XNA_count', 'previous_loans_NAME_PRODUCT_TYPE_XNA_count', 'previous_loans_NAME_PRODUCT_TYPE_walk-in_count', 'previous_loans_NAME_PRODUCT_TYPE_walk-in_count_norm', 'previous_loans_NAME_PRODUCT_TYPE_x-sell_count_norm', 'previous_loans_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_loans_CHANNEL_TYPE_AP+ (Cash loan)_count_norm', 'previous_loans_CHANNEL_TYPE_Car dealer_count', 'previous_loans_CHANNEL_TYPE_Car dealer_count_norm', 'previous_loans_CHANNEL_TYPE_Channel of corporate sales_count', 'previous_loans_CHANNEL_TYPE_Channel of corporate sales_count_norm', 'previous_loans_CHANNEL_TYPE_Contact center_count', 'previous_loans_CHANNEL_TYPE_Contact center_count_norm', 'previous_loans_CHANNEL_TYPE_Country-wide_count', 'previous_loans_CHANNEL_TYPE_Credit and cash offices_count', 'previous_loans_CHANNEL_TYPE_Credit and cash offices_count_norm', 'previous_loans_CHANNEL_TYPE_Regional / Local_count', 'previous_loans_CHANNEL_TYPE_Regional / Local_count_norm', 'previous_loans_CHANNEL_TYPE_Stone_count', 'previous_loans_CHANNEL_TYPE_Stone_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Auto technology_count', 'previous_loans_NAME_SELLER_INDUSTRY_Auto technology_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Clothing_count', 'previous_loans_NAME_SELLER_INDUSTRY_Clothing_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Connectivity_count', 'previous_loans_NAME_SELLER_INDUSTRY_Connectivity_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Consumer electronics_count', 'previous_loans_NAME_SELLER_INDUSTRY_Consumer electronics_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Furniture_count', 'previous_loans_NAME_SELLER_INDUSTRY_Furniture_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Industry_count', 'previous_loans_NAME_SELLER_INDUSTRY_Industry_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Jewelry_count', 'previous_loans_NAME_SELLER_INDUSTRY_Jewelry_count_norm', 'previous_loans_NAME_SELLER_INDUSTRY_Tourism_count', 'previous_loans_NAME_YIELD_GROUP_XNA_count', 'previous_loans_NAME_YIELD_GROUP_high_count', 'previous_loans_NAME_YIELD_GROUP_high_count_norm', 'previous_loans_NAME_YIELD_GROUP_low_action_count', 'previous_loans_NAME_YIELD_GROUP_low_action_count_norm', 'previous_loans_NAME_YIELD_GROUP_low_normal_count', 'previous_loans_NAME_YIELD_GROUP_low_normal_count_norm', 'previous_loans_NAME_YIELD_GROUP_middle_count', 'previous_loans_NAME_YIELD_GROUP_middle_count_norm', 'previous_loans_PRODUCT_COMBINATION_Card Street_count', 'previous_loans_PRODUCT_COMBINATION_Card Street_count_norm', 'previous_loans_PRODUCT_COMBINATION_Card X-Sell_count', 'previous_loans_PRODUCT_COMBINATION_Card X-Sell_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash_count', 'previous_loans_PRODUCT_COMBINATION_Cash_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash Street: high_count', 'previous_loans_PRODUCT_COMBINATION_Cash Street: high_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash Street: low_count', 'previous_loans_PRODUCT_COMBINATION_Cash Street: low_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash Street: middle_count', 'previous_loans_PRODUCT_COMBINATION_Cash Street: middle_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash X-Sell: high_count', 'previous_loans_PRODUCT_COMBINATION_Cash X-Sell: low_count', 'previous_loans_PRODUCT_COMBINATION_Cash X-Sell: low_count_norm', 'previous_loans_PRODUCT_COMBINATION_Cash X-Sell: middle_count', 'previous_loans_PRODUCT_COMBINATION_Cash X-Sell: middle_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS household with interest_count', 'previous_loans_PRODUCT_COMBINATION_POS household without interest_count', 'previous_loans_PRODUCT_COMBINATION_POS household without interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS industry with interest_count', 'previous_loans_PRODUCT_COMBINATION_POS industry with interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS industry without interest_count', 'previous_loans_PRODUCT_COMBINATION_POS industry without interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS mobile with interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS mobile without interest_count', 'previous_loans_PRODUCT_COMBINATION_POS mobile without interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS other with interest_count', 'previous_loans_PRODUCT_COMBINATION_POS other with interest_count_norm', 'previous_loans_PRODUCT_COMBINATION_POS others without interest_count', 'previous_loans_PRODUCT_COMBINATION_POS others without interest_count_norm', 'previous_loans_AMT_ANNUITY_mean', 'previous_loans_AMT_ANNUITY_max', 'previous_loans_AMT_ANNUITY_min', 'previous_loans_AMT_ANNUITY_sum', 'previous_loans_AMT_APPLICATION_mean', 'previous_loans_AMT_APPLICATION_max', 'previous_loans_AMT_APPLICATION_min', 'previous_loans_AMT_APPLICATION_sum', 'previous_loans_AMT_CREDIT_mean', 'previous_loans_AMT_CREDIT_max', 'previous_loans_AMT_CREDIT_min', 'previous_loans_AMT_CREDIT_sum', 'previous_loans_AMT_DOWN_PAYMENT_mean', 'previous_loans_AMT_DOWN_PAYMENT_max', 'previous_loans_AMT_DOWN_PAYMENT_min', 'previous_loans_AMT_DOWN_PAYMENT_sum', 'previous_loans_AMT_GOODS_PRICE_mean', 'previous_loans_AMT_GOODS_PRICE_max', 'previous_loans_AMT_GOODS_PRICE_min', 'previous_loans_AMT_GOODS_PRICE_sum', 'previous_loans_HOUR_APPR_PROCESS_START_mean', 'previous_loans_HOUR_APPR_PROCESS_START_max', 'previous_loans_HOUR_APPR_PROCESS_START_min', 'previous_loans_HOUR_APPR_PROCESS_START_sum', 'previous_loans_NFLAG_LAST_APPL_IN_DAY_sum', 'previous_loans_RATE_DOWN_PAYMENT_mean', 'previous_loans_RATE_DOWN_PAYMENT_max', 'previous_loans_RATE_DOWN_PAYMENT_sum', 'previous_loans_RATE_INTEREST_PRIVILEGED_sum', 'previous_loans_SELLERPLACE_AREA_mean', 'previous_loans_SELLERPLACE_AREA_max', 'previous_loans_SELLERPLACE_AREA_min', 'previous_loans_SELLERPLACE_AREA_sum', 'previous_loans_CNT_PAYMENT_mean', 'previous_loans_CNT_PAYMENT_max', 'previous_loans_CNT_PAYMENT_min', 'previous_loans_CNT_PAYMENT_sum', 'previous_loans_DAYS_FIRST_DRAWING_mean', 'previous_loans_DAYS_FIRST_DRAWING_max', 'previous_loans_DAYS_FIRST_DRAWING_min', 'previous_loans_DAYS_FIRST_DRAWING_sum', 'previous_loans_DAYS_FIRST_DUE_mean', 'previous_loans_DAYS_FIRST_DUE_max', 'previous_loans_DAYS_FIRST_DUE_min', 'previous_loans_DAYS_FIRST_DUE_sum', 'previous_loans_DAYS_LAST_DUE_1ST_VERSION_mean', 'previous_loans_DAYS_LAST_DUE_1ST_VERSION_max', 'previous_loans_DAYS_LAST_DUE_1ST_VERSION_min', 'previous_loans_DAYS_LAST_DUE_1ST_VERSION_sum', 'previous_loans_DAYS_LAST_DUE_mean', 'previous_loans_DAYS_LAST_DUE_max', 'previous_loans_DAYS_LAST_DUE_min', 'previous_loans_DAYS_LAST_DUE_sum', 'previous_loans_DAYS_TERMINATION_mean', 'previous_loans_DAYS_TERMINATION_max', 'previous_loans_DAYS_TERMINATION_min', 'previous_loans_DAYS_TERMINATION_sum', 'previous_loans_NFLAG_INSURED_ON_APPROVAL_mean', 'previous_loans_NFLAG_INSURED_ON_APPROVAL_min', 'previous_loans_counts', 'client_cash_NAME_CONTRACT_STATUS_Active_count_mean', 'client_cash_NAME_CONTRACT_STATUS_Active_count_max', 'client_cash_NAME_CONTRACT_STATUS_Active_count_min', 'client_cash_NAME_CONTRACT_STATUS_Active_count_sum', 'client_cash_NAME_CONTRACT_STATUS_Active_count_norm_sum', 'client_cash_NAME_CONTRACT_STATUS_Approved_count_min', 'client_cash_NAME_CONTRACT_STATUS_Approved_count_sum', 'client_cash_NAME_CONTRACT_STATUS_Approved_count_norm_min', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_mean', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_max', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_min', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_sum', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_norm_mean', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_norm_max', 'client_cash_NAME_CONTRACT_STATUS_Completed_count_norm_sum', 'client_cash_NAME_CONTRACT_STATUS_Returned to the store_count_mean', 'client_cash_NAME_CONTRACT_STATUS_Returned to the store_count_min', 'client_cash_NAME_CONTRACT_STATUS_Returned to the store_count_sum', 'client_cash_NAME_CONTRACT_STATUS_Returned to the store_count_norm_max', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_mean', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_max', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_min', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_sum', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_norm_mean', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_norm_max', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_norm_min', 'client_cash_NAME_CONTRACT_STATUS_Signed_count_norm_sum', 'client_cash_SK_ID_CURR_mean_mean', 'client_cash_SK_ID_CURR_mean_max', 'client_cash_SK_ID_CURR_mean_min', 'client_cash_SK_ID_CURR_mean_sum', 'client_cash_SK_ID_CURR_max_mean', 'client_cash_SK_ID_CURR_max_max', 'client_cash_SK_ID_CURR_max_min', 'client_cash_SK_ID_CURR_max_sum', 'client_cash_SK_ID_CURR_min_mean', 'client_cash_SK_ID_CURR_min_max', 'client_cash_SK_ID_CURR_min_min', 'client_cash_SK_ID_CURR_min_sum', 'client_cash_SK_ID_CURR_sum_mean', 'client_cash_SK_ID_CURR_sum_max', 'client_cash_SK_ID_CURR_sum_min', 'client_cash_SK_ID_CURR_sum_sum', 'client_cash_CNT_INSTALMENT_mean_mean', 'client_cash_CNT_INSTALMENT_mean_max', 'client_cash_CNT_INSTALMENT_mean_min', 'client_cash_CNT_INSTALMENT_mean_sum', 'client_cash_CNT_INSTALMENT_max_mean', 'client_cash_CNT_INSTALMENT_max_max', 'client_cash_CNT_INSTALMENT_max_min', 'client_cash_CNT_INSTALMENT_max_sum', 'client_cash_CNT_INSTALMENT_min_mean', 'client_cash_CNT_INSTALMENT_min_max', 'client_cash_CNT_INSTALMENT_min_min', 'client_cash_CNT_INSTALMENT_min_sum', 'client_cash_CNT_INSTALMENT_sum_mean', 'client_cash_CNT_INSTALMENT_sum_max', 'client_cash_CNT_INSTALMENT_sum_min', 'client_cash_CNT_INSTALMENT_sum_sum', 'client_cash_CNT_INSTALMENT_FUTURE_mean_mean', 'client_cash_CNT_INSTALMENT_FUTURE_mean_max', 'client_cash_CNT_INSTALMENT_FUTURE_mean_min', 'client_cash_CNT_INSTALMENT_FUTURE_mean_sum', 'client_cash_CNT_INSTALMENT_FUTURE_max_mean', 'client_cash_CNT_INSTALMENT_FUTURE_max_max', 'client_cash_CNT_INSTALMENT_FUTURE_max_min', 'client_cash_CNT_INSTALMENT_FUTURE_max_sum', 'client_cash_CNT_INSTALMENT_FUTURE_min_mean', 'client_cash_CNT_INSTALMENT_FUTURE_min_max', 'client_cash_CNT_INSTALMENT_FUTURE_min_min', 'client_cash_CNT_INSTALMENT_FUTURE_min_sum', 'client_cash_CNT_INSTALMENT_FUTURE_sum_mean', 'client_cash_CNT_INSTALMENT_FUTURE_sum_max', 'client_cash_CNT_INSTALMENT_FUTURE_sum_min', 'client_cash_CNT_INSTALMENT_FUTURE_sum_sum', 'client_cash_SK_DPD_mean_mean', 'client_cash_SK_DPD_mean_max', 'client_cash_SK_DPD_mean_min', 'client_cash_SK_DPD_mean_sum', 'client_cash_SK_DPD_max_mean', 'client_cash_SK_DPD_max_max', 'client_cash_SK_DPD_max_min', 'client_cash_SK_DPD_max_sum', 'client_cash_SK_DPD_min_mean', 'client_cash_SK_DPD_min_max', 'client_cash_SK_DPD_min_sum', 'client_cash_SK_DPD_sum_mean', 'client_cash_SK_DPD_sum_max', 'client_cash_SK_DPD_sum_min', 'client_cash_SK_DPD_sum_sum', 'client_cash_SK_DPD_DEF_mean_mean', 'client_cash_SK_DPD_DEF_mean_max', 'client_cash_SK_DPD_DEF_mean_min', 'client_cash_SK_DPD_DEF_mean_sum', 'client_cash_SK_DPD_DEF_max_mean', 'client_cash_SK_DPD_DEF_max_max', 'client_cash_SK_DPD_DEF_max_min', 'client_cash_SK_DPD_DEF_max_sum', 'client_cash_SK_DPD_DEF_sum_mean', 'client_cash_SK_DPD_DEF_sum_max', 'client_cash_SK_DPD_DEF_sum_min', 'client_cash_SK_DPD_DEF_sum_sum', 'client_cash_counts_mean', 'client_cash_counts_min', 'client_cash_counts_sum', 'client_counts_x', 'client_credit_NAME_CONTRACT_STATUS_Active_count_mean', 'client_credit_NAME_CONTRACT_STATUS_Active_count_max', 'client_credit_NAME_CONTRACT_STATUS_Active_count_min', 'client_credit_NAME_CONTRACT_STATUS_Active_count_sum', 'client_credit_NAME_CONTRACT_STATUS_Active_count_norm_sum', 'client_credit_NAME_CONTRACT_STATUS_Completed_count_mean', 'client_credit_NAME_CONTRACT_STATUS_Completed_count_max', 'client_credit_NAME_CONTRACT_STATUS_Completed_count_min', 'client_credit_NAME_CONTRACT_STATUS_Completed_count_sum', 'client_credit_NAME_CONTRACT_STATUS_Completed_count_norm_sum', 'client_credit_NAME_CONTRACT_STATUS_Sent proposal_count_sum', 'client_credit_NAME_CONTRACT_STATUS_Signed_count_sum', 'client_credit_SK_ID_CURR_mean_sum', 'client_credit_SK_ID_CURR_max_sum', 'client_credit_SK_ID_CURR_min_sum', 'client_credit_SK_ID_CURR_sum_mean', 'client_credit_SK_ID_CURR_sum_max', 'client_credit_SK_ID_CURR_sum_min', 'client_credit_SK_ID_CURR_sum_sum', 'client_credit_AMT_BALANCE_mean_mean', 'client_credit_AMT_BALANCE_mean_max', 'client_credit_AMT_BALANCE_mean_min', 'client_credit_AMT_BALANCE_mean_sum', 'client_credit_AMT_BALANCE_max_mean', 'client_credit_AMT_BALANCE_max_max', 'client_credit_AMT_BALANCE_max_min', 'client_credit_AMT_BALANCE_max_sum', 'client_credit_AMT_BALANCE_min_mean', 'client_credit_AMT_BALANCE_min_max', 'client_credit_AMT_BALANCE_min_min', 'client_credit_AMT_BALANCE_min_sum', 'client_credit_AMT_BALANCE_sum_mean', 'client_credit_AMT_BALANCE_sum_max', 'client_credit_AMT_BALANCE_sum_min', 'client_credit_AMT_BALANCE_sum_sum', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_mean_max', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_mean_min', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_mean_sum', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_max_mean', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_max_max', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_max_min', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_max_sum', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_min_mean', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_min_max', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_min_min', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_min_sum', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_sum_mean', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_sum_max', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_sum_min', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_sum_sum', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_max', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_min', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_sum', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_max_mean', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_max_max', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_max_min', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_max_sum', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_min_mean', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_min_max', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_min_min', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_min_sum', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_sum_mean', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_sum_max', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_sum_min', 'client_credit_AMT_DRAWINGS_ATM_CURRENT_sum_sum', 'client_credit_AMT_DRAWINGS_CURRENT_mean_mean', 'client_credit_AMT_DRAWINGS_CURRENT_mean_max', 'client_credit_AMT_DRAWINGS_CURRENT_mean_min', 'client_credit_AMT_DRAWINGS_CURRENT_mean_sum', 'client_credit_AMT_DRAWINGS_CURRENT_max_mean', 'client_credit_AMT_DRAWINGS_CURRENT_max_max', 'client_credit_AMT_DRAWINGS_CURRENT_max_min', 'client_credit_AMT_DRAWINGS_CURRENT_max_sum', 'client_credit_AMT_DRAWINGS_CURRENT_min_mean', 'client_credit_AMT_DRAWINGS_CURRENT_min_max', 'client_credit_AMT_DRAWINGS_CURRENT_min_min', 'client_credit_AMT_DRAWINGS_CURRENT_min_sum', 'client_credit_AMT_DRAWINGS_CURRENT_sum_mean', 'client_credit_AMT_DRAWINGS_CURRENT_sum_max', 'client_credit_AMT_DRAWINGS_CURRENT_sum_min', 'client_credit_AMT_DRAWINGS_CURRENT_sum_sum', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_mean_max', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_mean_min', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_mean_sum', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_max_mean', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_max_max', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_max_min', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_max_sum', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_sum_mean', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_sum_max', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_sum_min', 'client_credit_AMT_DRAWINGS_OTHER_CURRENT_sum_sum', 'client_credit_AMT_DRAWINGS_POS_CURRENT_mean_mean', 'client_credit_AMT_DRAWINGS_POS_CURRENT_mean_max', 'client_credit_AMT_DRAWINGS_POS_CURRENT_mean_min', 'client_credit_AMT_DRAWINGS_POS_CURRENT_mean_sum', 'client_credit_AMT_DRAWINGS_POS_CURRENT_max_mean', 'client_credit_AMT_DRAWINGS_POS_CURRENT_max_max', 'client_credit_AMT_DRAWINGS_POS_CURRENT_max_min', 'client_credit_AMT_DRAWINGS_POS_CURRENT_max_sum', 'client_credit_AMT_DRAWINGS_POS_CURRENT_min_mean', 'client_credit_AMT_DRAWINGS_POS_CURRENT_min_max', 'client_credit_AMT_DRAWINGS_POS_CURRENT_min_min', 'client_credit_AMT_DRAWINGS_POS_CURRENT_min_sum', 'client_credit_AMT_DRAWINGS_POS_CURRENT_sum_mean', 'client_credit_AMT_DRAWINGS_POS_CURRENT_sum_max', 'client_credit_AMT_DRAWINGS_POS_CURRENT_sum_min', 'client_credit_AMT_DRAWINGS_POS_CURRENT_sum_sum', 'client_credit_AMT_INST_MIN_REGULARITY_mean_mean', 'client_credit_AMT_INST_MIN_REGULARITY_mean_max', 'client_credit_AMT_INST_MIN_REGULARITY_mean_min', 'client_credit_AMT_INST_MIN_REGULARITY_mean_sum', 'client_credit_AMT_INST_MIN_REGULARITY_max_mean', 'client_credit_AMT_INST_MIN_REGULARITY_max_max', 'client_credit_AMT_INST_MIN_REGULARITY_max_sum', 'client_credit_AMT_INST_MIN_REGULARITY_min_mean', 'client_credit_AMT_INST_MIN_REGULARITY_min_max', 'client_credit_AMT_INST_MIN_REGULARITY_min_min', 'client_credit_AMT_INST_MIN_REGULARITY_min_sum', 'client_credit_AMT_INST_MIN_REGULARITY_sum_mean', 'client_credit_AMT_INST_MIN_REGULARITY_sum_max', 'client_credit_AMT_INST_MIN_REGULARITY_sum_min', 'client_credit_AMT_INST_MIN_REGULARITY_sum_sum', 'client_credit_AMT_PAYMENT_CURRENT_mean_mean', 'client_credit_AMT_PAYMENT_CURRENT_mean_max', 'client_credit_AMT_PAYMENT_CURRENT_mean_min', 'client_credit_AMT_PAYMENT_CURRENT_mean_sum', 'client_credit_AMT_PAYMENT_CURRENT_max_mean', 'client_credit_AMT_PAYMENT_CURRENT_max_max', 'client_credit_AMT_PAYMENT_CURRENT_max_min', 'client_credit_AMT_PAYMENT_CURRENT_max_sum', 'client_credit_AMT_PAYMENT_CURRENT_min_mean', 'client_credit_AMT_PAYMENT_CURRENT_min_max', 'client_credit_AMT_PAYMENT_CURRENT_min_min', 'client_credit_AMT_PAYMENT_CURRENT_min_sum', 'client_credit_AMT_PAYMENT_CURRENT_sum_mean', 'client_credit_AMT_PAYMENT_CURRENT_sum_max', 'client_credit_AMT_PAYMENT_CURRENT_sum_min', 'client_credit_AMT_PAYMENT_CURRENT_sum_sum', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_mean_max', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_mean_min', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_mean_sum', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_max_mean', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_max_max', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_max_min', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_max_sum', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_min_mean', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_min_max', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_min_min', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_min_sum', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_sum_mean', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_sum_max', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_sum_min', 'client_credit_AMT_PAYMENT_TOTAL_CURRENT_sum_sum', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_mean_max', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_mean_min', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_mean_sum', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_max_mean', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_max_max', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_max_min', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_max_sum', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_min_mean', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_min_max', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_min_min', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_min_sum', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_sum_mean', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_sum_max', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_sum_min', 'client_credit_AMT_RECEIVABLE_PRINCIPAL_sum_sum', 'client_credit_AMT_RECIVABLE_mean_mean', 'client_credit_AMT_RECIVABLE_mean_max', 'client_credit_AMT_RECIVABLE_mean_min', 'client_credit_AMT_RECIVABLE_mean_sum', 'client_credit_AMT_RECIVABLE_max_mean', 'client_credit_AMT_RECIVABLE_max_max', 'client_credit_AMT_RECIVABLE_max_min', 'client_credit_AMT_RECIVABLE_max_sum', 'client_credit_AMT_RECIVABLE_min_mean', 'client_credit_AMT_RECIVABLE_min_max', 'client_credit_AMT_RECIVABLE_min_min', 'client_credit_AMT_RECIVABLE_min_sum', 'client_credit_AMT_RECIVABLE_sum_mean', 'client_credit_AMT_RECIVABLE_sum_max', 'client_credit_AMT_RECIVABLE_sum_min', 'client_credit_AMT_RECIVABLE_sum_sum', 'client_credit_AMT_TOTAL_RECEIVABLE_mean_mean', 'client_credit_AMT_TOTAL_RECEIVABLE_mean_max', 'client_credit_AMT_TOTAL_RECEIVABLE_mean_min', 'client_credit_AMT_TOTAL_RECEIVABLE_mean_sum', 'client_credit_AMT_TOTAL_RECEIVABLE_max_mean', 'client_credit_AMT_TOTAL_RECEIVABLE_max_max', 'client_credit_AMT_TOTAL_RECEIVABLE_max_min', 'client_credit_AMT_TOTAL_RECEIVABLE_max_sum', 'client_credit_AMT_TOTAL_RECEIVABLE_min_mean', 'client_credit_AMT_TOTAL_RECEIVABLE_min_max', 'client_credit_AMT_TOTAL_RECEIVABLE_min_min', 'client_credit_AMT_TOTAL_RECEIVABLE_min_sum', 'client_credit_AMT_TOTAL_RECEIVABLE_sum_mean', 'client_credit_AMT_TOTAL_RECEIVABLE_sum_max', 'client_credit_AMT_TOTAL_RECEIVABLE_sum_min', 'client_credit_AMT_TOTAL_RECEIVABLE_sum_sum', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_mean_sum', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_max_mean', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_max_max', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_max_min', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_max_sum', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_min_sum', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_sum_mean', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_sum_max', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_sum_min', 'client_credit_CNT_DRAWINGS_ATM_CURRENT_sum_sum', 'client_credit_CNT_DRAWINGS_CURRENT_mean_mean', 'client_credit_CNT_DRAWINGS_CURRENT_mean_max', 'client_credit_CNT_DRAWINGS_CURRENT_mean_min', 'client_credit_CNT_DRAWINGS_CURRENT_max_mean', 'client_credit_CNT_DRAWINGS_CURRENT_max_max', 'client_credit_CNT_DRAWINGS_CURRENT_max_min', 'client_credit_CNT_DRAWINGS_CURRENT_max_sum', 'client_credit_CNT_DRAWINGS_CURRENT_min_mean', 'client_credit_CNT_DRAWINGS_CURRENT_min_max', 'client_credit_CNT_DRAWINGS_CURRENT_min_min', 'client_credit_CNT_DRAWINGS_CURRENT_min_sum', 'client_credit_CNT_DRAWINGS_CURRENT_sum_min', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_mean_sum', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_max_sum', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_sum_mean', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_sum_max', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_sum_min', 'client_credit_CNT_DRAWINGS_OTHER_CURRENT_sum_sum', 'client_credit_CNT_DRAWINGS_POS_CURRENT_mean_mean', 'client_credit_CNT_DRAWINGS_POS_CURRENT_mean_max', 'client_credit_CNT_DRAWINGS_POS_CURRENT_mean_min', 'client_credit_CNT_DRAWINGS_POS_CURRENT_mean_sum', 'client_credit_CNT_DRAWINGS_POS_CURRENT_max_sum', 'client_credit_CNT_DRAWINGS_POS_CURRENT_min_mean', 'client_credit_CNT_DRAWINGS_POS_CURRENT_min_max', 'client_credit_CNT_DRAWINGS_POS_CURRENT_min_min', 'client_credit_CNT_DRAWINGS_POS_CURRENT_min_sum', 'client_credit_CNT_DRAWINGS_POS_CURRENT_sum_mean', 'client_credit_CNT_DRAWINGS_POS_CURRENT_sum_max', 'client_credit_CNT_DRAWINGS_POS_CURRENT_sum_min', 'client_credit_CNT_DRAWINGS_POS_CURRENT_sum_sum', 'client_credit_CNT_INSTALMENT_MATURE_CUM_mean_mean', 'client_credit_CNT_INSTALMENT_MATURE_CUM_mean_max', 'client_credit_CNT_INSTALMENT_MATURE_CUM_mean_min', 'client_credit_CNT_INSTALMENT_MATURE_CUM_mean_sum', 'client_credit_CNT_INSTALMENT_MATURE_CUM_max_mean', 'client_credit_CNT_INSTALMENT_MATURE_CUM_max_max', 'client_credit_CNT_INSTALMENT_MATURE_CUM_max_min', 'client_credit_CNT_INSTALMENT_MATURE_CUM_max_sum', 'client_credit_CNT_INSTALMENT_MATURE_CUM_min_sum', 'client_credit_CNT_INSTALMENT_MATURE_CUM_sum_mean', 'client_credit_CNT_INSTALMENT_MATURE_CUM_sum_max', 'client_credit_CNT_INSTALMENT_MATURE_CUM_sum_min', 'client_credit_CNT_INSTALMENT_MATURE_CUM_sum_sum', 'client_credit_SK_DPD_mean_mean', 'client_credit_SK_DPD_mean_max', 'client_credit_SK_DPD_mean_min', 'client_credit_SK_DPD_mean_sum', 'client_credit_SK_DPD_max_mean', 'client_credit_SK_DPD_max_max', 'client_credit_SK_DPD_max_min', 'client_credit_SK_DPD_max_sum', 'client_credit_SK_DPD_sum_mean', 'client_credit_SK_DPD_sum_max', 'client_credit_SK_DPD_sum_min', 'client_credit_SK_DPD_sum_sum', 'client_credit_SK_DPD_DEF_mean_mean', 'client_credit_SK_DPD_DEF_mean_max', 'client_credit_SK_DPD_DEF_mean_min', 'client_credit_SK_DPD_DEF_mean_sum', 'client_credit_SK_DPD_DEF_max_mean', 'client_credit_SK_DPD_DEF_max_max', 'client_credit_SK_DPD_DEF_max_min', 'client_credit_SK_DPD_DEF_max_sum', 'client_credit_SK_DPD_DEF_sum_mean', 'client_credit_SK_DPD_DEF_sum_max', 'client_credit_SK_DPD_DEF_sum_min', 'client_credit_SK_DPD_DEF_sum_sum', 'client_credit_counts_mean', 'client_credit_counts_max', 'client_credit_counts_min', 'client_credit_counts_sum', 'client_counts_y', 'client_installments_SK_ID_CURR_mean_mean', 'client_installments_SK_ID_CURR_mean_max', 'client_installments_SK_ID_CURR_mean_min', 'client_installments_SK_ID_CURR_mean_sum', 'client_installments_SK_ID_CURR_max_mean', 'client_installments_SK_ID_CURR_max_max', 'client_installments_SK_ID_CURR_max_min', 'client_installments_SK_ID_CURR_max_sum', 'client_installments_SK_ID_CURR_min_mean', 'client_installments_SK_ID_CURR_min_max', 'client_installments_SK_ID_CURR_min_min', 'client_installments_SK_ID_CURR_min_sum', 'client_installments_SK_ID_CURR_sum_mean', 'client_installments_SK_ID_CURR_sum_max', 'client_installments_SK_ID_CURR_sum_min', 'client_installments_SK_ID_CURR_sum_sum', 'client_installments_NUM_INSTALMENT_VERSION_mean_max', 'client_installments_NUM_INSTALMENT_VERSION_mean_min', 'client_installments_NUM_INSTALMENT_VERSION_mean_sum', 'client_installments_NUM_INSTALMENT_VERSION_max_max', 'client_installments_NUM_INSTALMENT_VERSION_max_min', 'client_installments_NUM_INSTALMENT_VERSION_max_sum', 'client_installments_NUM_INSTALMENT_VERSION_min_mean', 'client_installments_NUM_INSTALMENT_VERSION_min_max', 'client_installments_NUM_INSTALMENT_VERSION_min_min', 'client_installments_NUM_INSTALMENT_VERSION_min_sum', 'client_installments_NUM_INSTALMENT_VERSION_sum_mean', 'client_installments_NUM_INSTALMENT_VERSION_sum_max', 'client_installments_NUM_INSTALMENT_VERSION_sum_min', 'client_installments_NUM_INSTALMENT_VERSION_sum_sum', 'client_installments_NUM_INSTALMENT_NUMBER_mean_mean', 'client_installments_NUM_INSTALMENT_NUMBER_mean_max', 'client_installments_NUM_INSTALMENT_NUMBER_mean_min', 'client_installments_NUM_INSTALMENT_NUMBER_mean_sum', 'client_installments_NUM_INSTALMENT_NUMBER_max_mean', 'client_installments_NUM_INSTALMENT_NUMBER_max_max', 'client_installments_NUM_INSTALMENT_NUMBER_max_min', 'client_installments_NUM_INSTALMENT_NUMBER_max_sum', 'client_installments_NUM_INSTALMENT_NUMBER_min_mean', 'client_installments_NUM_INSTALMENT_NUMBER_min_max', 'client_installments_NUM_INSTALMENT_NUMBER_min_sum', 'client_installments_NUM_INSTALMENT_NUMBER_sum_mean', 'client_installments_NUM_INSTALMENT_NUMBER_sum_max', 'client_installments_NUM_INSTALMENT_NUMBER_sum_min', 'client_installments_NUM_INSTALMENT_NUMBER_sum_sum', 'client_installments_AMT_INSTALMENT_mean_mean', 'client_installments_AMT_INSTALMENT_mean_max', 'client_installments_AMT_INSTALMENT_mean_min', 'client_installments_AMT_INSTALMENT_mean_sum', 'client_installments_AMT_INSTALMENT_max_mean', 'client_installments_AMT_INSTALMENT_max_max', 'client_installments_AMT_INSTALMENT_max_min', 'client_installments_AMT_INSTALMENT_max_sum', 'client_installments_AMT_INSTALMENT_min_mean', 'client_installments_AMT_INSTALMENT_min_max', 'client_installments_AMT_INSTALMENT_min_min', 'client_installments_AMT_INSTALMENT_min_sum', 'client_installments_AMT_INSTALMENT_sum_mean', 'client_installments_AMT_INSTALMENT_sum_max', 'client_installments_AMT_INSTALMENT_sum_min', 'client_installments_AMT_INSTALMENT_sum_sum', 'client_installments_AMT_PAYMENT_mean_mean', 'client_installments_AMT_PAYMENT_mean_max', 'client_installments_AMT_PAYMENT_mean_min', 'client_installments_AMT_PAYMENT_mean_sum', 'client_installments_AMT_PAYMENT_max_mean', 'client_installments_AMT_PAYMENT_max_max', 'client_installments_AMT_PAYMENT_max_min', 'client_installments_AMT_PAYMENT_max_sum', 'client_installments_AMT_PAYMENT_min_mean', 'client_installments_AMT_PAYMENT_min_max', 'client_installments_AMT_PAYMENT_min_min', 'client_installments_AMT_PAYMENT_min_sum', 'client_installments_AMT_PAYMENT_sum_mean', 'client_installments_AMT_PAYMENT_sum_max', 'client_installments_AMT_PAYMENT_sum_min', 'client_installments_AMT_PAYMENT_sum_sum', 'client_installments_counts_mean', 'client_installments_counts_max', 'client_installments_counts_min', 'client_installments_counts_sum', 'client_counts']



```python
chi2_selector = SelectKBest(chi2, k=5) # Select top 5 features
X_train_kbest = chi2_selector.fit_transform(X_train, y_train)

selected_features = [feature for feature, mask in zip(X_train.columns, chi2_selector.get_support()) if mask]
print(selected_features)
```

    ['client_credit_SK_ID_CURR_sum_sum', 'client_credit_AMT_CREDIT_LIMIT_ACTUAL_sum_sum', 'client_installments_SK_ID_CURR_sum_sum', 'client_installments_AMT_INSTALMENT_sum_sum', 'client_installments_AMT_PAYMENT_sum_sum']



## Information gain
This technique measures the reduction in entropy after we do something to the dataset. For example, if we want to split the data for a decision tree model, we would like maximum gain.

## Correlation coefficient
This is the indicator of correlation between target variable and the input variables, it selects the highly correlated features with the target.

# Wrapper method

## Forward selection
This starts with an empty model. In each step, it considers all the available model and chooses the one that add the most to the performance of the model. The process is repeated until adding more variables to the model doesn't improve it anymore.

## Backward elimination
This starts with a model full of features. It removes the least significant feature one at a time, until no further improvement is observed.

## Recursive feature elimination (RFE)
This algorithm performs a greedy search to find the best performing feature subset. In each iteration, it creates model and determine the best and worst features. Then it builds the next model with the best ones until the end. The features are then ranked based on the order of their elimination.



```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3) # Select top 3 features
fit = rfe.fit(X_train, y_train)

# To get the selected feature names
selected_features = [feature for feature, mask in zip(X_train.columns, rfe.support_) if mask]
print(selected_features)

```


```python

```


```python

```


```python

```


```python

```


# Embedded method
This method chooses the features as a part of the creation of the model.

## LASSO 
LASSO prefers fewer non zero coefficients.



```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5) 
lasso.fit(X_train, y_train)

# To get the selected feature names
selected_features = [feature for feature, coef in zip(X_train.columns, lasso.coef_) if coef != 0]
print(selected_features)

```


## Elastic net

## Decision tree

A decision uses Gini or entropy impurity to estimate the quality of each split. Features at the top contribute to the final prediction more than others at the leaf. The expected fraction of the samples they contribute to thus can be used as an indicator of their importance.




```python

```
