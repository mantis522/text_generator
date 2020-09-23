
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

augmented_train_df = pd.concat(([pos_generation, neg_generation])).reset_index(drop=True)
origin_train_df = making_origin_df(home_origin_dir)
# removed_train_df = pd.concat([removed_neg_PP, removed_pos_PP]).reset_index(drop=True)
concat_train_df = pd.concat([augmented_train_df, origin_train_df]).reset_index(drop=True)

print(concat_train_df)