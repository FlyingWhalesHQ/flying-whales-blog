---
layout: post
title:  "Recurrent Neural Network"
date:   2023-03-15 10:14:54 +0700
categories: jekyll update
---


# Introduction

Remember the threshold unit in the perceptron? A recurrent neuron is a neuron that apart from that feedforward part, it has connections backward. This wires it for sequential data such as time series, natural language, but not exclusively. For small dataset, a fully connected network can do the trick, and for huge dataset, a convolutional net is capable, too. But how does a recurrent neural network work? The answer is simple, the backward connection wires output/activation at time t-1 to be a part of input in time t. So this neuron carries information from the past (but not the future). Plus, the weights for input and previous output are shared among neurons. For natural language processing, this can be understood as if the neural net can understand and carry context of each word.

In mathematical notations, let s be the combination of previous output and input, U, W, $$ \theta $$ are weights. Also, let use 30 neurons to output one final prediction, we would need to calculate the combination of input and previous output 30 times before we can activate the result:

$$ s_0 = 0 $$

$$ s_t = f(U * x_t + W * s_{t-1}, t >= 1 $$

$$ \hat{y} = h_{\theta} (\theta * s_{30}) $$

## Backpropagation through time

We propagate the loss back through the recurrent neurons as usual. Since those neurons are in time, this is called backpropagation throught time. We have 3 weight matrices: W,U, and $$ \theta$$. To descend the gradient, we need to find partial derivatives: $$ \frac{\partial Loss}{\partial U}, \frac{\partial Loss}{\partial V},\frac{\partial Loss}{\partial \theta} $$

$$ \frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \hat{y}}  \frac{\partial \hat{y}}{\partial \theta} $$

$$ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}}  \frac{\partial \hat{y}}{\partial s_{30}}  \frac{\partial s_{30}}{\partial W} $$

with $$ s_{30} = f(W * s_{29} + V * x_{30}) $$

so $$ \frac{s_{30}}{\partial W} = s'_{30}(W) + \frac{\partial s_{30}}{\partial s_{29}}  \frac{\partial s_{29}}{\partial W} $$

$$ \Rightarrow  \frac{\partial L}{\partial W} = \sum_{i=0}^{30}  \frac{\partial L}{\partial \hat{y}}   \frac{\partial \hat{y}}{\partial s_{30}}  \frac{\partial s_{30}}{\partial s_i} + s'_i(W) $$ with $$ \frac{\partial s_{30}}{\partial s_i} = \prod_{j=i}^{29} \frac{\partial s_{j+1}}{\partial s_j} $$

# Code example

Let's use RNN for a time series dataset that is about electric production. Firstly, we simply plot the dataset with the index as the date. Second, we run a simple RNN neural net. After that, we use a LSTM model and then a GRU. LSTM and GRU are different variants of the RNN, they bring more memory into the cell and have gates to build/erase memory and patterns learned in the process.


```python
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("Electric_Production.csv",parse_dates=["DATE"])
df.columns = ["date", "amount"]  # shorter names
df = df.sort_values("date").set_index("date")
df = df.drop_duplicates()
df

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
      <th>amount</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-01-01</th>
      <td>72.5052</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>70.6720</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>62.4502</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>57.4714</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>55.3151</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-09-01</th>
      <td>98.6154</td>
    </tr>
    <tr>
      <th>2017-10-01</th>
      <td>93.6137</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>97.3359</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>114.7212</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>129.4048</td>
    </tr>
  </tbody>
</table>
<p>397 rows × 1 columns</p>
</div>




```python
df.plot(grid=True, marker=".", figsize=(8, 3.5))
plt.show()

```


    
![png](14RecurrentNet_files/14RecurrentNet_2_0.png)
    


![14RecurrentNet_2_0](https://user-images.githubusercontent.com/7457301/225576802-653c7e1b-1d85-4b9f-aa84-dc9eae1b6c5c.png)


```python
# Simple RNN
time_steps=15 # time steps is the number of data points used to predict the next one
# 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df2 = pd.read_csv('/kaggle/input/time-series-datasets/Electric_Production.csv')
df2.columns = ["date", "amount"]  # shorter names
df2 = df2.sort_values("date").set_index("date")
df2 = df2.drop_duplicates()
df2
train = df2[:250]
test=df2[250:]

training_data = train

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

x_training_data = []
y_training_data =[]

for i in range(time_steps, len(training_data)):
    x_training_data.append(training_data[i-time_steps:i, 0])
    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

print(x_training_data.shape)
print(y_training_data.shape)

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], 
                                               x_training_data.shape[1], 
                                               1))
print(x_training_data.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout

rnn = Sequential()
rnn.add(SimpleRNN(3, input_shape=(time_steps,1), activation="tanh"))
rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

test_data=test
print(test_data.shape)
plt.plot(test_data)

all_data=df2

x_test_data = all_data[len(all_data) - len(test_data) - time_steps:].values
x_test_data = np.reshape(x_test_data, (-1, 1))
x_test_data = scaler.transform(x_test_data)

final_x_test_data = []

for i in range(time_steps, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-time_steps:i, 0])
final_x_test_data = np.array(final_x_test_data)

final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], 
                                               final_x_test_data.shape[1], 
                                               1))
predictions = rnn.predict(final_x_test_data)

plt.clf() #This clears the old plot from our canvas

plt.plot(predictions)

unscaled_predictions = scaler.inverse_transform(predictions)
plt.clf() #This clears the first prediction plot from our canvas

plt.plot(unscaled_predictions)
plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
plt.plot(test_data, color = 'black', label = "Real Data")
plt.title('Electric Production Predictions')

```

Epoch 100/100
8/8 [==============================] - 0s 5ms/step - loss: 0.0376

![rnn](https://user-images.githubusercontent.com/7457301/225573952-2d3dc1ae-73f6-416e-a097-099dd49445a5.png)

## LSTM - Long short term memory cell

To address the issue of the recurrent neuron in which it doesn't remember too long in the past, long short term memory neuron was architected. Apart from the current input, there is a thread of long term memory which carries recognized pattern. There is also a thread of short term memory which carries the information from the previous time step. Inside the cell, there are gates to control the flow of data. Since those gates use sigmoid function, they output 1 to open the gate and 0 to close the gate. There are usually 3 gates: forget gate, input gate and output gate. Those gates choose which pattern to forget, which new one to take into the long term memory thread, and it also controls how much of long term data should be used in this cell to output immediately.

Here is a LSTM net for the electric production dataset:


```python
# 3
rnn = Sequential()
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

```

Epoch 100/100
8/8 [==============================] - 0s 38ms/step - loss: 0.0193

![lstm](https://user-images.githubusercontent.com/7457301/225573947-f589aa41-1170-47e6-8e46-3a95ce9b1e0c.png)

Here is a LSTM that predicts the next 14 days of eletric production:


```python
rnn = Sequential()
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))
rnn.add(Dense(units = 14))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

```

Epoch 100/100
8/8 [==============================] - 0s 38ms/step - loss: 0.0182

![lstm14](https://user-images.githubusercontent.com/7457301/225573937-8aa08490-719f-447b-90c2-928673bf4324.png)

### Time distributed layer

In LSTM, when you return_sequences=True (predict a sequence instead of a single point in time), technically the next layer needs to be able to input a sequence of values. If you want to cover the LSTM with a Dense layer on top, you need to turn off return_sequence. There is a way to handle the sequence (result of the hidden layers) returned, using a Time Distribution layer. That time distributed layer would receive all the sequence returned by LSTM and process to return whatever a usual Dense layer would return.

Here is a net with time distributed wrapped around LSTM:



```python
rnn = Sequential()
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))
for i in [True, True, True]:
    rnn.add(LSTM(units = 45, return_sequences = i))
rnn.add(TimeDistributed(Dense(units=1)))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

```

Epoch 100/100
8/8 [==============================] - 0s 30ms/step - loss: 0.0469

## Bidirectional neural net

Since LSTM runs only from the past to the present, there is an extension that called Bidirectional LSTM in which we wire the future back to the past. This is to mimic the situation in speaking where we think in advance til the end and only then we formulate a sentence. Since there are information at the end of that sentence that needs to be thought through before we know which words to use for the beginning of the sentence.

Processing natural language is messy. There is a step before we can run the model: to encode words into digital formats (vector of numbers), this vector is also called the vector representation of the word. After that step, we can apply all the matrix multiplications. Since the words are translated into numbers, we can calculate the sentiment of a paragraph, to see whether it is a positive or negative review. Here is one example recipe, to classify sentiment of IMDB reviews:

- Load dataset from tensorflow, split the train and test sets

- Encode the text (turn words into tokens, store the results in vectors)

- Add a bidirectional layer

- Add a fully connected layer, for the classification part

- Compile and train the model with usual favorite optimizer and loss

- Plot the train and test errors

Result: When the training loss is low and validation loss is high, the model overfits. We achieve reasonable accuracy though.


```python
import tensorflow_datasets as tfds
dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
train = train_dataset.take(4000)
test = test_dataset.take(1000)
# to shuffle the data ...
BUFFER_SIZE = 4000 # we will put all the data into this big buffer, and sample randomly from the buffer
BATCH_SIZE = 128  # we will read 128 reviews at a time

train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test = test.batch(BATCH_SIZE)
train = train.prefetch(BUFFER_SIZE)
test = test.prefetch(BUFFER_SIZE)

VOCAB_SIZE=1000 # assuming our vocabulary is just 1000 words

encoder = layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)

encoder.adapt(train.map(lambda text, label: text)) # we just encode the text, not the labels

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

example, label = list(train.take(1))[0] # that's one batch
len(example)
example[0].numpy()
encoded_example = encoder(example[:1]).numpy()
encoded_example

model = tf.keras.Sequential([
    encoder, # the encoder
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to ignore the padding 0 (reviews get padded to be equal in len)
        mask_zero=True),
    tf.keras.layers.Bidirectional(layers.LSTM(64)), # making LSTM bidirectional
    tf.keras.layers.Dense(32, activation='relu'), # FC layer for the classification part
    tf.keras.layers.Dense(1) # final FC layer

])

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions[0])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # adam optimizer is more efficient (not always the most accurate though)
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

model.summary()

H2 = model.fit(train, epochs=25, validation_data=test)

def plot_results(H,name):
    results = pd.DataFrame({"Train Loss": H.history['loss'], "Validation Loss": H.history['val_loss'],
              "Train Accuracy": H.history['accuracy'], "Validation Accuracy": H.history['val_accuracy']
             })
    fig, ax = plt.subplots(nrows=2, figsize=(16, 9))
    results[["Train Loss", "Validation Loss"]].plot(ax=ax[0])
    results[["Train Accuracy", "Validation Accuracy"]].plot(ax=ax[1])
    ax[0].set_xlabel("Epoch")
    ax[1].set_xlabel("Epoch")
    plt.show()
    plt.savefig(name)
    

plot_results(H2,'imdb')
```

![imdb](https://user-images.githubusercontent.com/7457301/225586057-ca991bf3-0464-46c8-ba14-4e51698611aa.png)

## GRU - Gated recurrent unit

A GRU is similar to LSTM cell in which it has gates and a thread of memory. There is one gate to control which part of data should be recalled from the memory and which part should use the current input. It is a simplified LSTM. Follows is a GRU net for electric production data:


```python
rnn = Sequential()
rnn.add(GRU(256, return_sequences=False))# 3
rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

```

Epoch 100/100
8/8 [==============================] - 0s 30ms/step - loss: 0.0114

![gru](https://user-images.githubusercontent.com/7457301/225573951-fe95fd90-06e5-4940-96ab-1a1d3784ef2c.png)

GRU is the one that gives the best loss: 0.014
