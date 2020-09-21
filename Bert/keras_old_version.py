import matplotlib.pyplot as plt
import os
import tarfile
from six.moves import urllib
import glob
import numpy as np
import codecs
import sys
# The Keras utils used:
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
from keras.initializers import TruncatedNormal
import bert
from bert import tokenization
from bert.tokenization import FullTokenizer
from keras_bert import AdamWarmup
from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint


# IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# IMDB_PATH = os.path.join("data")
#
#
# def fetch_data(url=IMDB_URL, path=IMDB_PATH):
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     tgz_path = os.path.join(path, "aclImdb_v1.tar.gz")
#     urllib.request.urlretrieve(url, tgz_path)
#     data_tgz = tarfile.open(tgz_path)
#     data_tgz.extractall(path = path)
#     data_tgz.close()


def read_txt_file(path):
    with open(path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()
        text = " ".join(lines)
    return text


def retrieve_data(data_type="train"):
    imdb_path = os.path.join(os.getcwd(), "data/aclImdb/")
    current_path = os.path.join(imdb_path, data_type)
    x = []
    y = []
    for sent_type in ["pos", "neg"]:
        sent_path = os.path.join(current_path, sent_type, "*.txt")
        sent_paths = glob.glob(sent_path)
        x += [read_txt_file(path) for path in sent_paths]
        if sent_type == "pos":
            y += [1.0] * len(sent_paths)
        else:
            y += [0.0] * len(sent_paths)

    return x, y


# If data is not loaded into datasets/aclImdb - it will create the directory and fetch the data:
# if 'aclImdb' not in os.listdir():
#     print("***** IMDB data not loaded - Fetching data from source *****")
#     fetch_data()

# Load the data:
x_train, y_train = retrieve_data()
x_test, y_test = retrieve_data(data_type = "test")

print(x_train[0])
print(y_train[0])


print(len(x_train))
print(type(x_train))
print(len(y_train))
print(type(y_train))

# Shuffle the data independently:
train_idx = np.random.randint(0, len(x_train), len(x_train))
test_idx = np.random.randint(0, len(x_test), len(x_test))

x_train, y_train = [x_train[index] for index in train_idx], [y_train[index] for index in train_idx]
x_test, y_test = [x_test[index] for index in test_idx], [y_test[index] for index in test_idx]

# If model is not installed and unzipped - it will be:
BERT_PRETRAINED_DIR = 'uncased_L-12_H-768_A-12'

if BERT_PRETRAINED_DIR not in os.listdir():
    print("***** Loading pretrained BERT model to directory: {} *****".format(os.path.join(os.getcwd(),BERT_PRETRAINED_DIR)))


# The chosen parameters provide good performance and will work on a Colab GPU runtime:
learning_rate = 2e-5
weight_decay = 0.001
epochs = 3
batch_size = 16
max_seq_len = 256

## step parameter - For custom AdamOptimizer for BERT finetuning
decay_steps = int(epochs*len(x_train)/batch_size)
warmup_steps = int(0.1*decay_steps)

print("Number of LR decay steps: {0} \nNumber of warm-up steps: {1}".format(decay_steps, warmup_steps))

# Next we read the BERT model that we just loaded:
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True,seq_len = max_seq_len)
print("Lookup model architecture with: bert_model.summary()")
print("I dare ya'")

# Initialize custom Adam optimizer with warmup:
adam_warmup = AdamWarmup(lr=learning_rate, decay_steps = decay_steps,
                      warmup_steps = warmup_steps, weight_decay = weight_decay)

# Picking BERT layers and building output layers:
input_layer = bert_model.input
embedding_output  = bert_model.layers[-6].output
output_layer = Dense(1, activation='sigmoid',kernel_initializer=TruncatedNormal(stddev=0.02),name = 'class_output')(embedding_output)
model  = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer=adam_warmup, metrics = ["acc"])
model.summary()

# Functions for converting text to BERT token ids:

# Whenever texts is stored in an numpy array:
def convert_lines_from_array(example, max_seq_length,tokenizer):
    max_seq_length -=2     # Cause we append [CLS] and [SEP]
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a)>max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
        longer += 1
      one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
      all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

# Whenever texts is stored in a list:
def convert_lines_from_list(example, max_seq_length,tokenizer):
    max_seq_length -=2     # Cause we append [CLS] and [SEP]
    all_tokens = []
    longer = 0
    for i in range(len(example)):
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a)>max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
        longer += 1
      one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
      all_tokens.append(one_token)
    print("Number of texts with more BERT tokens than max_seq_len: {0}".format(longer))
    return np.array(all_tokens)

# Get path to BERT model vocab file:
vocab_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

# Instantiate BERT repository tokenizer:
tokenizer = FullTokenizer(vocab_file = vocab_path, do_lower_case = True) # lowercases text (NB. we use uncased model)

# Creates arrays of BERT tokens ids, segment ids (0's) and masking ids (1's):
token_input = convert_lines_from_list(x_train, max_seq_len, tokenizer)
seg_input = np.zeros(token_input.shape)
mask_input = np.ones(token_input.shape)

model.fit([token_input, seg_input, mask_input], y_train, batch_size = batch_size, epochs = epochs)


# Convert test text to Token ids, segment ids and masking ids:
test_token_input = convert_lines_from_list(x_test, max_seq_len, tokenizer)
test_seg_input = np.zeros(test_token_input.shape)
test_mask_input = np.ones(test_token_input.shape)
print("Tokenization of test set done")

# Evaluating on test set
predict = model.evaluate([test_token_input, test_seg_input, test_mask_input], y_test, batch_size = batch_size)
print(predict)