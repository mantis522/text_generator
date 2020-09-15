from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
import time
from nltk.tokenize import sent_tokenize, word_tokenize
import tensorflow as tf
import re

start = time.time()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

text= "Vijay Krishna Acharyas Tashan is a overhyped, stylized, product. Sure its a one of the most stylish films, but when it comes to content, even the masses will reject this one. Why? The films script is as amateur as a 2 year old baby. Script is king, without a good script even the greatest director of alltime cannot do anything. Tashan is produced by the most successful production banner Yash Raj Films and Mega Stars appearing in it. But nothing on earth can save you if you script is bland. Thumbs down! Performances Anil Kapoor, is a veteran actor. But how could he okay a role like this? Akshay Kumar is great actor, in fact he is the sole saving grace. Kareena Kapoor has never looked so hot. She looks stunning and leaves you, all stand up. Saif Ali Khan does not get his due in here. Sanjay Mishra, Manoj Phawa and Yashpal Sharma are wasted.Tashan is a boring film. The films failure at the box office, should you keep away."
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# encode context the generation is conditioned on
input_ids = tokenizer.encode(text, return_tensors='tf')

cur_len = shape_list(input_ids)[1]

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=cur_len + 40)

# print("Output:\n" + 100 * '-')
output_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
output_text = " ".join(output_text.split())
print(output_text)

print(sent_tokenize(output_text)[1])
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
print(cur_len)
# beam_output = model.generate(
#     input_ids,
#     max_length=text_word_len+20,
#     num_beams=5,
#     no_repeat_ngram_size=2,
#     early_stopping=True
# )
#
# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

print("time : ", time.time() - start)