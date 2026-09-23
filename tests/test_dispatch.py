"""Tests host custom-autograd dispatch using a fake extension, not CUDA kernel execution."""
import torch
import torch.nn.functional as F
from budgetce import ops

class FakeExtension:
    def row_loss(self,z,y,ignore):
        return F.cross_entropy(z.float(),y,ignore_index=ignore,reduction='none')
    def backward_inplace(self,z,y,scale,ignore):
        p=torch.softmax(z.float(),1);good=y!=ignore;safe=y.masked_fill(~good,0)
        p.scatter_add_(1,safe[:,None],-torch.ones(len(y),1))
        p.mul_(good[:,None]).mul_(scale);z.copy_(p)

def test_cuda_dispatch_algebra_with_fake_extension(monkeypatch):
    monkeypatch.setattr(ops,'load_extension',lambda:FakeExtension())
    torch.manual_seed(2)
    h=torch.randn(8,5,requires_grad=True);w=torch.randn(11,5,requires_grad=True);y=torch.arange(8);y[2]=-100
    ref=ops.native_loss(h,w,y);gr=torch.autograd.grad(ref*128,(h,w))
    out=ops._Chunked.apply(h,w,y,3,'cuda',torch.float32,-100,'mean')
    gg=torch.autograd.grad(out*128,(h,w))
    torch.testing.assert_close(out,ref)
    for a,b in zip(gr,gg):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
