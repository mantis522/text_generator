import json

with open("D:/data/data_augmentation2/pos/EX.json", encoding='utf-8', errors='ignore') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['augmented_text']

print(len(json_string))


# for a in range(len(json_string)):
#     print(json_string[a])