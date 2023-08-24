import os
import tensorflow as tf
import gpt_2_simple as gpt2
import re

gpt2.download_gpt2(model_name='124M')

file_path = "gpt2.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    # Convertir a minúsculas
    text = text.lower()
    return text

preprocessed_text = preprocess_text(text)

preprocessed_file_path = "TestoToken"
with open(preprocessed_file_path, 'w', encoding='utf-8') as file:
    file.write(preprocessed_text)

sess = gpt2.start_tf_sess()

if os.path.isdir(os.path.join("checkpoint", "run1")):
    gpt2.load_gpt2(sess, run_name='run1')
else:
    gpt2.finetune(sess,
                  dataset=preprocessed_file_path,
                  model_name='124M',
                  steps=500,
                  restore_from='fresh',
                  run_name='run1',
                  print_every=10,
                  sample_every=200,
                  save_every=500
                  )

while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        break
    else:
        preprocessed_input = preprocess_text(user_input)
        generated_text = gpt2.generate(sess, model_name='124M', prefix=preprocessed_input, length=75, temperature=0.9, return_as_list=True)
        filtered_text = [response for response in generated_text if response != preprocessed_input]
        if filtered_text:
            print("IA:", filtered_text[0])
        else:
            print("IA: No tengo una respuesta adecuada en este momento.")

gpt2.save_gpt2(sess, run_name='run1')