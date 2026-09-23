import torch
from budgetce.model import TinyDecoder
from budgetce.ops import native_loss,chunked_loss

def test_model_gradients():
    torch.manual_seed(6)
    m=TinyDecoder(vocab=31,d=16,layers=1,heads=2,max_seq=8).double()
    x=torch.randint(31,(2,9));ys=x[:,1:].contiguous().flatten()
    g=[]
    for method in ['native','torch']:
        m.zero_grad(set_to_none=True);h=m(x[:,:-1].contiguous())
        loss=native_loss(h,m.classifier,ys) if method=='native' else chunked_loss(h,m.classifier,ys,3)
        loss.backward();g.append([p.grad.clone() for p in m.parameters()])
    for a,b in zip(*g):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)

def test_model_causality():
    torch.manual_seed(8);m=TinyDecoder(vocab=31,d=16,layers=1,heads=2,max_seq=8).eval()
    x=torch.randint(31,(1,8));y=x.clone();y[0,-1]=(y[0,-1]+1)%31
    torch.testing.assert_close(m(x)[:-1],m(y)[:-1])
