import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import re
import json
import random

home_origin_dir = "D:/ruin/data/json_data/train_data_full.json"
home_test_dir = "D:/ruin/data/json_data/test_data_full.json"

home_RPPN_directory = "D:/ruin/data/json_data/removed_data/removed_PP_neg.json"
home_RRPP_directory = "D:/ruin/data/json_data/removed_data/removed_PP_pos.json"

pos_gen_dir = "D:/ruin/data/gen_txt/pos_generate.txt"
pos_gen_sentence_dir = "D:/ruin/data/gen_txt/pos_sentence.txt"
neg_gen_dir = "D:/ruin/data/gen_txt/neg_generate.txt"
neg_gen_sentence_dir = "D:/ruin/data/gen_txt/neg_sentence.txt"

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

def making_df(file_directory, label):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    removed_sentence = json_data['removed_sentence']
    removed_sentence = removed_sentence[0]
    removed = []

    for a in range(len(removed_sentence)):
        for b in range(len(removed_sentence[a])):
            removed.append(removed_sentence[a][b])

    df = pd.DataFrame(removed)
    df.columns = ['data']
    df['label'] = label

    return df

pos_generation = making_augmented_df(pos_gen_sentence_dir, 1)
neg_generation = making_augmented_df(neg_gen_sentence_dir, 0)

origin_train_df = making_origin_df(home_origin_dir)
test_df = making_test_df(home_test_dir)
test_df = test_df.sample(frac=1).reset_index(drop=True)
removed_neg_PP = making_df(home_RPPN_directory, 0)
removed_pos_PP = making_df(home_RRPP_directory, 1)

augmented_train_df = pd.concat(([pos_generation, neg_generation])).reset_index(drop=True)

# removed_train_df = pd.concat([removed_neg_PP, removed_pos_PP]).reset_index(drop=True)
concat_train_df = pd.concat([augmented_train_df, origin_train_df]).reset_index(drop=True)

TAG_RE = re.compile(r'<[^>]+>')


def clean_text(sentence):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = TAG_RE.sub('', sentence)

    return sentence

concat_train_df['clean_reviews'] = concat_train_df['data'].astype(str).apply(clean_text)
test_df['clean_reviews'] = test_df['data'].astype(str).apply(clean_text)

y_train = concat_train_df['label']
y_test = test_df['label']

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

tokenized_reviews = [tokenize_reviews(data) for data in concat_train_df.data]
reviews_with_len = [[review, y_train[i], len(review)]
                 for i, review in enumerate(tokenized_reviews)]

tokenized_reviews2 = [tokenize_reviews(data) for data in test_df.data]
reviews_with_len2 = [[review, y_test[i], len(review)]
                 for i, review in enumerate(tokenized_reviews2)]


random.shuffle(reviews_with_len)

reviews_with_len.sort(key=lambda x: x[2])
reviews_with_len2.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
sorted_reviews_labels2 = [(review_lab2[0], review_lab2[1]) for review_lab2 in reviews_with_len2]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))
processed_dataset2 = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels2, output_types=(tf.int32, tf.int32))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
batched_dataset2 = processed_dataset2.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

train_data = batched_dataset
test_data = batched_dataset2

class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 5

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="nadam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="nadam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_data, epochs=NB_EPOCHS, batch_size=64, verbose=1)


score, acc = text_model.evaluate(test_data)
print('Test score : ', score)
print('Test acc : ', acc)
