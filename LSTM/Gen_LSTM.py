import json
import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
from matplotlib import pyplot as plt

start_time = time.time()

origin_dir = "D:/data/json_data/train_data_full.json"
test_dir = "D:/data/json_data/test_data_full.json"

pos_gen_dir = "../pos_generate.txt"
pos_gen_sentence_dir = "../pos_sentence.txt"
neg_gen_dir = "../neg_generate.txt"
neg_gen_sentence_dir = "../neg_sentence.txt"

def making_augmented_df(file_dir, labels):
    aug_list = []
    f = open(file_dir, 'r', encoding='utf-8')
    data = f.read()
    data = data.rsplit('\n')
    for a in range(len(data)):
        if data[a] != '':
            aug_list.append(data[a])
        else:
            pass
    f.close()

    aug_list = set(aug_list)

    df_aug = pd.DataFrame(aug_list, columns=['data'])
    df_aug['label'] = labels
    # df_aug = df_aug.sample(frac=1).reset_index(drop=True)
    # df_aug = df_aug[:3000]


    return df_aug

# c = making_augmented_df('C:/Users/ruin/PycharmProjects/text_generator/neg_generate.txt', 0)
# print(c)
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

pos_generation = making_augmented_df(pos_gen_dir, 1)
neg_generation = making_augmented_df(neg_gen_dir, 0)

origin_train_df = making_origin_df(origin_dir)
test_df = making_test_df(test_dir)
test_df = test_df.sample(frac=1).reset_index(drop=True)

augmented_train_df = pd.concat(([pos_generation, neg_generation])).reset_index(drop=True)

concat_train_df = pd.concat([augmented_train_df, origin_train_df]).reset_index(drop=True)

x_train = concat_train_df['data'].values
y_train = concat_train_df['label'].values

val_df = test_df[:12500]
test_df = test_df[12500:]

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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('Test score : ', score)
print('Test accuracy : ', acc)

print("--- %s seconds ---" % (time.time() - start_time))