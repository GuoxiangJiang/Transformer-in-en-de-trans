import torch 
from torch import nn
import torch.nn.functional as F
import math

from torch import Tensor
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super().__init__(vocab_size,d_model,padding_idx=1)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super().__init__()
        self.encoding = torch.zeros(max_len,d_model,device = device)
        self.encoding.requires_grad = False
        pos = torch.arange(0,max_len,device = device)
        pos = pos.float().unsqueeze(dim = 1)
        _2i = torch.arange(0,d_model,2,device = device).float()
        self.encoding[:,0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:,1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self,x):
        batch_size,seq_len = x.size()
        return self.encoding[:seq_len,:]

class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_len,device)
        self.dropout = nn.Dropout(drop_prob)
    def forward(self,x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_encoding(x)
        output_embedding = self.dropout(token_embedding + positional_embedding)
        return output_embedding

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
    
    def scaled_dot_product_attention(self,q,k,v,mask=None):
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores,dim = -1)
        output = torch.matmul(attention,v)
        return output,attention

    def split_heads(self,x):
        x = x.view(x.size(0),-1,self.n_head,self.d_k)
        return x.transpose(1,2)

    def combine_heads(self,x):
        x = x.transpose(1,2).contiguous().view(x.size(0),-1,self.d_model)
        return x

    def forward(self,q,k,v,mask = None):
       q = self.w_q(q)
       k = self.w_k(k)
       v = self.w_v(v)

       q = self.split_heads(q)
       k = self.split_heads(k)
       v = self.split_heads(v)

       output,attention = self.scaled_dot_product_attention(q,k,v,mask)
       output = self.combine_heads(output)
       output = self.w_o(output)
       return output,attention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(d_model, d_ff, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask=None):
        # Self attention
        _x = x
        x, _ = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(d_model, d_ff, drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
    
    def forward(self, dec, enc, src_mask=None, tgt_mask=None):
        # Self attention
        _x = dec
        x, _ = self.self_attention(dec, dec, dec, tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # Cross attention
        if enc is not None:
            _x = x
            x, _ = self.cross_attention(x, enc, enc, src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        
        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, drop_prob, max_len, device):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, drop_prob, max_len, device):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, dec, enc, src_mask=None, tgt_mask=None):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, src_mask, tgt_mask)
        output = self.linear(dec)
        return output

class Transformer(nn.Module):                                                                                                                                    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8, d_ff=2048, 
                 n_layers=6, drop_prob=0.1, max_len=512, device='cuda'):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_ff, n_layers, drop_prob, max_len, device)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_ff, n_layers, drop_prob, max_len, device)
        self.device = device
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output
    
    def make_src_mask(self, src):
        # src: (batch_size, src_len)
        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)  # 1 is pad token
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # tgt: (batch_size, tgt_len)
        tgt_pad_mask = (tgt != 1).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

