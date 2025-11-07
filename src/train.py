import os
# 设置镜像站点（必须在导入其他库之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from utils.tokenizer import get_tokenizer
from models.model import Transformer
import json
from torch.utils.tensorboard import SummaryWriter


# 数据集类
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # iwslt2017数据格式: {'translation': {'en': '...', 'de': '...'}}
        en_text = self.data[idx]['translation']['en']
        de_text = self.data[idx]['translation']['de']
        
        # MarianTokenizer编码（自动添加special tokens）
        src = self.tokenizer.encode(en_text, max_length=self.max_len, truncation=True)
        tgt = self.tokenizer.encode(de_text, max_length=self.max_len, truncation=True)
        
        return torch.LongTensor(src), torch.LongTensor(tgt)

# Padding函数
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=1)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=1)
    return src_padded, tgt_padded

# 训练函数
def train_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
        
        # 创建mask
        src_mask = model.make_src_mask(src)
        tgt_mask = model.make_tgt_mask(tgt_input)
        
        # 前向传播
        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录每个batch的loss
        global_step = epoch * len(loader) + i
        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
    
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    return avg_loss

# 验证函数
def evaluate(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            src_mask = model.make_src_mask(src)
            tgt_mask = model.make_tgt_mask(tgt_input)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
    return avg_loss

# 主函数
def main():
    # 配置
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 100
    max_len = 128
    
    # 加载数据和分词器
    print('加载数据集和分词器...')
    dataset = load_from_disk('./data/iwslt2017_dataset')
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 创建数据加载器
    train_dataset = TranslationDataset(dataset['train'], tokenizer, max_len)
    val_dataset = TranslationDataset(dataset['validation'], tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # 创建模型
    print(f'创建模型 (vocab_size={vocab_size}, device={device})...')
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        n_head=8,
        d_ff=1024,
        n_layers=3,
        drop_prob=0.1,
        max_len=max_len,
        device=device
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # 忽略pad token
    
    # 训练
    print('开始训练...')
    os.makedirs('../results', exist_ok=True)
    
    # 创建日志记录器
    
    writer = SummaryWriter(log_dir='../runs/translation')

    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        val_loss = evaluate(model, val_loader, criterion, device, writer, epoch)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 记录学习率
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '../results/best_model.pt')
            print(f'  保存最佳模型 (val_loss={val_loss:.4f})')

    writer.close()


if __name__ == '__main__':
    main()

