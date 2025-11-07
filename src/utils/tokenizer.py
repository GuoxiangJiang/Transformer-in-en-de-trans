import os
from transformers import MarianTokenizer

def get_tokenizer():
    # 设置镜像站点
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    return MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

def tokenize(text, tokenizer):
    return tokenizer.encode(text)

