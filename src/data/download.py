import os
from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de")
dataset.save_to_disk("./iwslt2017_dataset")