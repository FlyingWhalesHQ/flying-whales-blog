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



```python

```
