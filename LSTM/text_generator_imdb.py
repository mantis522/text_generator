import json
import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from keras import optimizers
from tensorflow.keras import preprocessing
from keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from nltk.tokenize import sent_tokenize
import tensorflow as tf
from tqdm import tqdm
import time

start_time = time.time()

# origin_neg_directory = "C:/Users/ruin/Desktop/data/json_data/train_neg_full.json"
# origin_pos_directory = "C:/Users/ruin/Desktop/data/json_data/train_pos_full.json"
origin_directory = "D:/data/train_data_full.json"
test_directory = "D:/data/test_data_full.json"

home_origin_dir = "D:/data/json_data/train_data_full.json"
home_test_dir = "D:/data/json_data/test_data_full.json"



def making_origin_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    train_review = []
    train_label = []

    train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train = pd.DataFrame(train_review, columns=['data'])
    df_train['label'] = train_label

    return df_train

def making_test_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    test_data = json_data['data']
    test_review = []
    test_label = []

    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])

    df = pd.DataFrame(test_review, columns=['data'])
    df['label'] = test_label

    return df

origin_train_df = making_origin_df(origin_directory)
test_df = making_test_df(test_directory)
test_df = test_df.sample(frac=1).reset_index(drop=True)

val_df = test_df[:12500]
test_df = test_df[12500:]

origin_train_df = pd.concat([origin_train_df] * 1, ignore_index=True)

x_train = origin_train_df['data'].values
y_train = origin_train_df['label'].values

x_val = val_df['data'].values
y_val = val_df['label'].values

x_test = test_df['data'].values
y_test = test_df['label'].values

vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

maxlen = 80


x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape)
print(x_val.shape)
print(y_train)
print(y_train.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=64, verbose=2, callbacks=[es])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('Test score : ', score)
print('Test accuracy : ', acc)

# model.save('sentiment_imdb_model.h5')


def index(word):
    if word in tokenizer.word_index:
        return tokenizer.word_index[word]
    else:
        return "0"

def sequences(words):
    words = text_to_word_sequence(words)
    seqs = [[index(word) for word in words if word != "0"]]
    return preprocessing.sequence.pad_sequences(seqs, maxlen=maxlen)

def sentiment_classification(text):
    text = sequences(text)
    if model.predict(text)[0][0] > 0.60:
        return "positive"
    elif model.predict(text)[0][0] < 0.40:
        return "negative"
    else:
        return "None"

def about_symbol(text):
    text = text.replace(".", ". ")
    text = text.replace(". . .", ". ")

    return text

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt_tokenizer.eos_token_id)

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
        before_input = json_string[a][:b]
        poham_input = json_string[a][:b+1]
        after_input = json_string[a][b+1:]
        # print("input 포함 : " + ' '.join(before_input) + " " + input_text + " " + ' '.join(after_input))
        # print("단순 리스트만 : " + ' '.join(poham_input) + " " + ' '.join(after_input))
        # print(' '.join(before_input) + " " + input_text + " " + ' '.join(after_input))
        input_ids = gpt_tokenizer.encode(input_text, return_tensors='tf')
        cur_len = shape_list(input_ids)[1]
        greedy_output = model_gpt.generate(input_ids, max_length=cur_len + 35)
        output_text = gpt_tokenizer.decode(greedy_output[0], skip_special_tokens=True)
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


f = open("../neg_generate.txt", 'w', encoding='utf-8', errors='ignore')
for i in range(len(text_list)):
    data = text_list[i] + "\n"
    f.write(data)
f.close()

g = open("../neg_sentence.txt", 'w', encoding='utf-8', errors='ignore')
for i in range(len(output_list)):
    data = output_list[i] + "\n"
    g.write(data)
g.close()

print("time : ", time.time() - start)