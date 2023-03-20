---
layout: post
title:  "Transformer"
date:   2023-03-17 14:14:54 +0700
categories: jekyll update
---

## Encoder - decoder

To encode is to turn words/images into numbers (digitalize) so that we can do matrix multiplication. With number, we can also construct a data space in which closer data points means similar meaning in the real world. To decode is to turn those numbers back into words/images. Through encoder-decoder, words can be translated to another langauges, images can be worked on more efficiently.

# Attention layer

Remember the architect of recurrent neural network (RNN)? It is a neuron that wires the previous output to the next input. For translation the translation task in general, we need to output a sequence of words. We also need a context for those words, so that context is translated into context, too. Therefore an attention layer is added to the RNN. This attention layer can be called a context vector. As the name suggests, it carries the context, i.e. let each word know which other words it needs to pay attention to after lots of training together. This would make the output words understand its position. And words that are together would carry higher weights in attenting to each other. The calculation for the attention layer is as follows:

- First, at each step, we calculate score for each word in related to all each words (including itself), using dot product or cosine similarity. This results in numbers saying how much is a word related to another word. Of course a word pays the most attention to itself.

- Let the scores run through a softmax function to turn them into probability distribution of the attention weights.

- Combine the weights with hidden state result to have context vector

- Concatenate the context vector and the hidden state vector. Transform the concanated vector with a weight vector. Activate it via the tanh function. That gives us the attention vector.

# Transformer

Even though LSTMs are useful, transformer is the next generation translator in which they use attention layer only. 

## The encoder

- Words are turned into vector of numbers. This forms an input 
matrix with the number of rows to be the length of the sentence (each word is a row). The number of columns are the dimensions of the word vector. More dimensions mean we can represent more complex meaning of the words, but more expensive computation. Plus sometimes it is not necessary to be so complex. 

- Positional encoding: since transformer only uses attention, it doesn't use any of the sequence analysis provided by RNN or CNN (no recurrence or convolution), we can shuffle the sentence and it would still predict true. To give it the sense of positions for each word, they use sin and cos functions. The positional encoding vector then is added to the input embedding vector.

- Sublayer 1: Multi-headed attention

    - Stack the row vectors together and we have the matrix embeding of the sentence. Multiply the matrix with 3 matrices to have a result of 3 corresponding matrices: Q (query), K (key), and V (value). Here we have the digital representation of 3 versions of the original sentence in which we vaguely understand the purpose of the sentence (query matrix), the keys of the sentence and the value of the sentence (again, each word is one row).

    - For each word (for each row), we multiply query row with key row. The paper by Google suggests to divide this number to squareroot of dimensions of keys, in case the result becomes too big. This is the attention score for each word in relation with all each words. These scores are then normalized to sum up to 1. This softmax mutiplies with value with create a convex combination of value (or weighted value). This is the attention vector for each word. As usual, more dimensions of this vector means more meaning representation. 

    - We just finish the calculation for one head of the attention. Usually we do multiple head. At the end, we concanate those heads into a matrix. Take another matrix W, so that the output matrix has the same dimension as the input matrix.
    
    - Some dropout, for regularization
    
    - Residual connection: We add the resulting matrix to the original input x of this sub-layer, so that more original information contributing to the sub-layer output.
    
    - Normalization layer: substract mean and divided by standard deviation, so that the data is normalized and centered back to zero. This would show any trend/dispersion in the layer better.

- Sublayer 2: Fully connected layer

    - At that output, each new word still is a row. Now that output matrix would let each word to go through fully connectec neurons (with ReLU activation) so that the number of dimensions are kept. We have the final output of an encoder block at this point.
    
    - Dropout, residual connection and normalization for this sub layer.

-> We can repeat this process multiple time, to have multiple encoding block (or layer).



## The decoder

- The output embedding, starting with a token signifying the beginning of a sentence

- The positional encoding

- Masked multi-head attention
    
    - We mask any attention to the subsequence (future word) of that word to minus infinity. With this, the decoding process would go from left to right, like time, and there would be no use of future words since that would be cheating time.
    
    - Apart from that, we do the processing like the multi headed attention in the encoder block: transforming the encoder's output into K and V to feed into each decoder's block to predict each word. Til it predicts an end of sentence token. The query matrix Q would be created using the output of the layer precedes it.
    
- Linear and Softmax layer

    - With a vocabulary of N words, the output of the previous layer would be projected onto this vector of N. Then the projection vector goes through a softmax layer to give probability distribution over N words.
    
    - With this probability distribution over N word vector, we can choose each word by: simply take the word with the highest possibilities (greedy algorithm), randomize over the vector, beam search (take 2 highest words and continue to extra polate two translations -> choose the one with less error)


# Code example

The most two famous implementations of the transformer are GPT and BERT, with GPT remains an auto regressive model (masked the future: you only rely on previous words to predict the next one), and BERT being bidirectional (the attention is not masked). In the code example, we would use a pretrained GPT-2 and continue to train with a corpus of data science text, so that it can finish our sentence with technical notes. Then we deploy the model on huggingface.

```python
from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, pipeline, Trainer, TrainingArguments
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
pds_data=TextDataset(
    tokenizer=tokenizer,
    file_path='data/tech.txt',
    block_size=32
)
model=GPT2LMHeadModel.from_pretrained('gpt2')
pretrained_generator=pipeline(
    'text-generation', model=model, tokenizer='gpt2',
    config={'max_length':200,'do_sample':True, 'top_p':0.9,'temperature':0.7,'top_k':10} 
)
```


```python
for generated_sequence in pretrained_generator('An unsupervised learning algorightm is', num_return_sequences=2):
  print(generated_sequence['generated_text'])
```

Here is the sentence finishing before our training:

- An unsupervised learning algorightm is often described as a self-learning system used for identifying unsupervised algorithms in tasks that involve multiple objects. An unsupervised learning system uses its own processor to identify inputs in these task conditions that
An unsupervised learning algorightm is often employed to determine if a person is "attached" to her "likes" on Pinterest or Tumblr. A group of scientists believe this way will lead to information about an individual's mental state


```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,  # MLM is Masked Language Modelling (for BERT + auto-encoding tasks)
)

training_args=TrainingArguments(
    output_dir='./gpt2_pds',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=len(pds_data.examples),
    logging_steps=50,
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    save_strategy='epoch'
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=pds_data.examples[:int(len(pds_data.examples)*.8)],
    eval_dataset=pds_data.examples[int(len(pds_data.examples)*.8):]
)
trainer.evaluate()
trainer.train()
```

|Epoch|	Training Loss |	Validation Loss|
|--|--|--|
|1|	4.222500|	3.959039|
|2| 3.853700|	3.712693|
|3|	3.626700|	3.587901|



```python
tokenizer._tokenizer.save("tokenizer.json")
trainer.save_model()
```

Try the model [here deployed on huggingface](https://huggingface.co/ayaderaghul/datascience-style-completion?text=Machine+learning+is+the+practice+that)

<embed src="https://huggingface.co/ayaderaghul/datascience-style-completion?text=Machine+learning+is+the+practice+that" width="300" height="200">

Here is the sentence finishing after training:

- An unsupervised learning algorightm is that they can be designed to behave like simple classes in real-world scenarios. For example, a simple training program where they're trained to do four things:

Select the first two items of


- Machine learning is the practice that will help you become a better student and make the most of your career if you're willing to commit to this and to work hard to get better.

It's often true about education that the hardest thing in the
