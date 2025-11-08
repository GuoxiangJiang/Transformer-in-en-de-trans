import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_from_disk
from utils.tokenizer import get_tokenizer
from models.model import Transformer
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge import Rouge


def translate(model, tokenizer, text, device, max_len=128):
    """翻译单句"""
    model.eval()
    with torch.no_grad():
        src = torch.LongTensor(tokenizer.encode(text, max_length=max_len, truncation=True)).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src)
        enc_output = model.encoder(src, src_mask)
        
        tgt_tokens = [tokenizer.bos_token_id or 0]
        for _ in range(max_len):
            tgt = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt)
            output = model.decoder(tgt, enc_output, src_mask, tgt_mask)
            next_token = output[0, -1, :].argmax().item()
            pretoken = tokenizer.decode(next_token)
            if pretoken == tokenizer.eos_token or pretoken == tokenizer.pad_token:
                break
            
            tgt_tokens.append(next_token)
        
        return tokenizer.decode(tgt_tokens, skip_special_tokens=True)


def compute_metrics(predictions, references):
    """计算 BLEU 和 ROUGE"""
    # 过滤空翻译（用占位符替换）
    clean_preds = [p if p.strip(".") else "900" for p in predictions]
    clean_refs = [r if r.strip(".") else "900" for r in references]
    
    # BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(clean_preds, [[ref] for ref in clean_refs])
    
    # ROUGE
    rouge = Rouge()

    rouge_scores = rouge.get_scores(clean_preds, clean_refs, avg=True)
    
    
    return {
        'BLEU': bleu_score.score,
        'ROUGE-1': rouge_scores.get('rouge-1', {}).get('f', 0) * 100,
        'ROUGE-2': rouge_scores.get('rouge-2', {}).get('f', 0) * 100,
        'ROUGE-L': rouge_scores.get('rouge-l', {}).get('f', 0) * 100,
    }


def main():
    
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    max_len = 128
    
    # 加载数据和模型
    dataset = load_from_disk('./data/iwslt2017_dataset')
    test_data = dataset['test']
    
    tokenizer = get_tokenizer()
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_head=8,
        d_ff=1024,
        n_layers=3,
        drop_prob=0.1,
        max_len=max_len,
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load('../results/best_model.pt', map_location=device))
    
    # 翻译
    predictions, references = [], []
    num = 0
    for item in tqdm(test_data):
        pred = translate(model, tokenizer, item['translation']['en'], device, max_len)
        num = num+1
        # if num>=1000:
        #     break
        predictions.append(pred)
        references.append(item['translation']['de'])
    
    
    # 评估
    scores = compute_metrics(predictions, references)
    
    # 输出结果
    print('\n评估结果:')
    for metric, score in scores.items():
        print(f'{metric}: {score:.2f}')
    

if __name__ == '__main__':
    main()

