import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, GRU, BatchNormalization
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import math

df = pd.read_csv("D:/data/17_742210_compressed_Tweets.csv/Tweets.csv")
df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)

x = df['text']
y = df['airline_sentiment']

t = Tokenizer()
t.fit_on_texts(x)

vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(x)

def max_tweet():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

tweet_num = max_tweet()
maxlen = tweet_num
padded_X = pad_sequences(sequences, padding='post', maxlen=maxlen)

labels = to_categorical(np.asarray(y))

x_train, x_test, y_train, y_test = train_test_split(padded_X, labels, test_size=0.2, random_state=0)

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)

embeddings_index = {}
f = open("D:/data/glove.6B/glove.6B.100d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))

# fill in matrix
for word, i in t.word_index.items():  # dictionary
    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe
    if embedding_vector is not None:
        # add to matrix
        embedding_matrix[i] = embedding_vector # each row of matrix

embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                           input_length = tweet_num, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist_1 = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save('sentiment_model.h5')

def index(word):
    if word in t.word_index:
        return t.word_index[word]
    else:
        return "0"

def sequences(words):
    words = text_to_word_sequence(words)
    seqs = [[index(word) for word in words if word != "0"]]
    return preprocessing.sequence.pad_sequences(seqs, maxlen=maxlen)

def sentiment_classification(text):
    text = sequences(text)
    if max((model.predict(text))[0]) == model.predict(text)[0][0]:
        return "negative"
    elif max((model.predict(text))[0]) == model.predict(text)[0][1]:
        return "neutral"
    else:
        return "positive"

# print(sentiment_classification("I was born in March 5"))
# print(sentiment_classification("Return the maximum along a given axis."))
#
neutral_seq = sequences("I was so excited to see it.")
neutral_seq2 = sequences("I think it's a lot of things.")
print(model.predict(neutral_seq))
print(model.predict(neutral_seq2))


