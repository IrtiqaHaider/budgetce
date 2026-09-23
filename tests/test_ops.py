import pytest
import torch
from budgetce.ops import native_loss,chunked_loss,validate_inputs

def tensors(dtype=torch.float64,n=7,v=13,d=5):
    g=torch.Generator().manual_seed(9)
    h=torch.randn(n,d,generator=g,dtype=dtype,requires_grad=True)
    w=(torch.randn(v,d,generator=g,dtype=dtype)/d**.5).requires_grad_()
    y=torch.arange(n)%v
    return h,w,y

@pytest.mark.parametrize('chunk',[1,3,7,20])
@pytest.mark.parametrize('reduction',['mean','sum'])
def test_gradient_equivalence(chunk,reduction):
    h,w,y=tensors();y[1]=-100
    a=native_loss(h,w,y,reduction=reduction)
    ga=torch.autograd.grad(2.3*a,(h,w))
    b=chunked_loss(h,w,y,chunk,reduction=reduction)
    gb=torch.autograd.grad(2.3*b,(h,w))
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for x,z in zip(ga,gb): torch.testing.assert_close(x,z,rtol=1e-11,atol=1e-12)

def test_gradcheck():
    h,w,y=tensors(n=3,v=5,d=2)
    assert torch.autograd.gradcheck(lambda a,b:chunked_loss(a,b,y,2),(h,w),eps=1e-6,atol=1e-5)

def test_all_ignored_zero():
    h,w,y=tensors();y.fill_(-100)
    loss=chunked_loss(h,w,y,3)
    grads=torch.autograd.grad(loss,(h,w))
    assert loss.item()==0 and all(torch.count_nonzero(g)==0 for g in grads)
    assert native_loss(h,w,y).item()==0

def test_invalid_target():
    h,w,y=tensors();y[0]=100
    with pytest.raises(ValueError,match='Target'): chunked_loss(h,w,y)

def test_noncontiguous_rejected():
    h,w,y=tensors(); h=h.T.contiguous().T
    with pytest.raises(ValueError,match='Contiguous'): chunked_loss(h,w,y)

@pytest.mark.parametrize('chunk',[0,-1,True,1.5])
def test_bad_chunk(chunk):
    h,w,y=tensors()
    with pytest.raises(ValueError):chunked_loss(h,w,y,chunk)

def test_fp16_scaled_gradients():
    h,w,y=tensors(torch.float32,n=19,v=107,d=32)
    a=native_loss(h,w,y,dtype=torch.float16)
    ga=torch.autograd.grad(128*a,(h,w))
    b=chunked_loss(h,w,y,7,dtype=torch.float16)
    gb=torch.autograd.grad(128*b,(h,w))
    torch.testing.assert_close(a,b,rtol=.001,atol=.001)
    for x,z in zip(ga,gb):
        assert float((x-z).norm()/x.norm().clamp_min(1e-8))<.003

def test_only_hidden_grad():
    h,w,y=tensors();w=w.detach()
    loss=chunked_loss(h,w,y,3);loss.backward()
    assert h.grad is not None

def test_gradient_accumulation():
    h,w,y=tensors(n=8)
    ref=torch.autograd.grad(native_loss(h,w,y),(h,w))[1]
    for a in (0,4):
        (chunked_loss(h[a:a+4],w,y[a:a+4],2)/2).backward()
    torch.testing.assert_close(w.grad,ref)
