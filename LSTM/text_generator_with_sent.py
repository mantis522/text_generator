import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import json
import time
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

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

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

tweet_num = max_tweet()
maxlen = tweet_num
padded_X = pad_sequences(sequences, padding='post', maxlen=maxlen)

labels = to_categorical(np.asarray(y))

x_train, x_test, y_train, y_test = train_test_split(padded_X, labels, test_size=0.2, random_state=0)

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)

model = load_model('../sentiment_model.h5')


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

def about_symbol(text):
    text = text.replace(".", ". ")
    text = text.replace(". . .", ". ")

    return text

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

with open("D:/data/edited_data/train_neg_edit_full.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['splited_sentence']

seq_length = 1000
start = time.time()

text_list = []
output_list= []

for a in tqdm(range(len(json_string))):
    # print("-" * 100)
    for b in range(len(json_string[a])):
        input_text = json_string[a][b]
        # before_input = json_string[a][:b]
        poham_input = json_string[a][:b+1]
        after_input = json_string[a][b+1:]
        # print("input 포함 : " + ' '.join(before_input) + " " + input_text + " " + ' '.join(after_input))
        # print("단순 리스트만 : " + ' '.join(poham_input) + " " + ' '.join(after_input))
        # print(' '.join(before_input) + " " + input_text + " " + ' '.join(after_input))
        input_ids = tokenizer.encode(input_text, return_tensors='tf')
        cur_len = shape_list(input_ids)[1]
        greedy_output = model_gpt.generate(input_ids, max_length=cur_len + 35)
        output_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        output_text = " ".join(output_text.split())
        output_text = about_symbol(output_text)
        # print("오리지널 + 생성 : " + output_text)
        try:
            output_text = sent_tokenize(output_text)[1]
        except IndexError:
            pass
        # print("생성된 문장만 : " + output_text)
        # print("생성된 문장 감성분석 : "+sentiment_classification(output_text))
        if sentiment_classification(output_text) == 'negative':
            sum_text = ' '.join(poham_input) + " " + output_text + " " + ' '.join(after_input)
            text_list.append(sum_text)
            output_list.append(output_text)
        else:
            sum_text = ''
            output_text = ''
            text_list.append(sum_text)
            output_list.append(output_text)
        # print("-" * 30)

f = open("../neg_generate.txt", 'w', encoding='utf-8')
for i in range(len(text_list)):
    data = text_list[i] + "\n"
    f.write(data)
f.close()

g = open("../neg_sentence.txt", 'w', encoding='utf-8')
for i in range(len(output_list)):
    data = output_list[i] + "\n"
    g.write(data)
g.close()

# text = "This is a good, dark film that I highly recommend."
# text_word_len = len(text.split())
# input_ids = tokenizer.encode(text, return_tensors='tf')
# greedy_output = model_gpt.generate(input_ids, max_length=text_word_len+50)
# output_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
# output_text = " ".join(output_text.split())
# print(output_text)
# sent_text = sent_tokenize(output_text)[1]
#
# print(sentiment_classification(sent_text))
#
# if sentiment_classification(sent_text) == 'positive':
#     print("이건 긍정입니다.")
# else:
#     print("긍정이 아닙니다.")



print("time : ", time.time() - start)