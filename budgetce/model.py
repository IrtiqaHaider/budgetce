"""Small untied-weight causal decoder. Synthetic next-token training, not quality evaluation."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

class Block(nn.Module):
    def __init__(self,d,heads):
        super().__init__();self.heads=heads;self.d=d
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d,bias=False);self.proj=nn.Linear(d,d,bias=False)
        self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,4*d,bias=False);self.fc2=nn.Linear(4*d,d,bias=False)
    def forward(self,x):
        b,s,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(b,s,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        with sdpa_kernel(SDPBackend.MATH):
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        x=x+self.proj(y.transpose(1,2).contiguous().reshape(b,s,d))
        return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))

class TinyDecoder(nn.Module):
    def __init__(self,vocab=32768,d=256,layers=4,heads=4,max_seq=1024):
        super().__init__()
        self.emb=nn.Embedding(vocab,d);self.pos=nn.Embedding(max_seq,d)
        self.blocks=nn.ModuleList([Block(d,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)
        self.classifier=nn.Parameter(torch.empty(vocab,d))
        self.apply(self._init);nn.init.normal_(self.classifier,std=.02)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Embedding,nn.Linear)):nn.init.normal_(m.weight,std=.02)
    def forward(self,x):
        h=self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        for layer in self.blocks:h=layer(h)
        return self.norm(h).reshape(-1,h.shape[-1]).contiguous()
