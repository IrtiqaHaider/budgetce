import gc
import torch
from .common import MIB
from .model import TinyDecoder
from .ops import run_plan
from .measure import time_block


def training_trial(sequence,method,seed,cfg,planner):
    gc.collect();torch.cuda.empty_cache();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    b=cfg['training_batch'];accum=cfg['training_accumulation'];v=cfg['training_vocab'];d=cfg['training_d']
    shape=dict(n=b*sequence,v=v,d=d)
    if method=='adaptive':
        choice=planner.select(shape,int(cfg['operator_selection_budget_mib']*MIB))
        if choice is None:raise ValueError('No adaptive plan predicted feasible for the training loss operator.')
        plan=choice['plan']
    else:choice=None;plan='native' if method=='native' else f"torch:{cfg['torch_chunk']}"
    model=TinyDecoder(v,d,cfg['training_layers'],cfg['training_heads'],max(cfg['training_sequences'])).cuda()
    model.train()
    # Input bank is allocated and shifted BEFORE measurement. Exact inputs are paired across methods.
    generator=torch.Generator(device='cuda').manual_seed(seed+1881)
    total=cfg['training_warmup']+cfg['training_updates']
    raw=torch.randint(v,(total,accum,b,sequence+1),device='cuda',generator=generator)
    bank=[[(r[...,:-1].contiguous(),r[...,1:].contiguous().flatten()) for r in update] for update in raw]
    del raw
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['loss_scale'],growth_interval=1000000)
    state={'updates':0,'opt_steps':0,'losses':[]}
    hook=opt.register_step_post_hook(lambda *args:state.__setitem__('opt_steps',state['opt_steps']+1))
    def clear():opt.zero_grad(set_to_none=True)
    def step():
        opt.zero_grad(set_to_none=True)
        k=state['updates'];total_loss=torch.zeros((),device='cuda')
        for x,y in bank[k]:
            with torch.autocast('cuda',dtype=torch.float16):
                h=model(x)
                loss=run_plan(plan,h,model.classifier,y,dtype=torch.float16)/accum
            scaler.scale(loss).backward()
            total_loss.add_(loss.detach())
        scaler.step(opt);scaler.update()
        state['updates']+=1;state['losses'].append(total_loss)
    def validate():
        if state['opt_steps']!=state['updates']:
            raise RuntimeError(f"AMP skipped optimizer work: {state['opt_steps']}/{state['updates']} updates.")
        losses=torch.stack(state['losses'])
        if not bool(torch.isfinite(losses).all().item()):raise RuntimeError('Nonfinite training loss.')
        if any(not bool(torch.isfinite(p).all().item()) for p in model.parameters()):
            raise RuntimeError('Nonfinite final parameters.')
    result=time_block(step,clear,validate,cfg['training_warmup'],cfg['training_updates'])
    total_tokens=b*sequence*accum*cfg['training_updates']
    result.update(plan=plan,selection=choice,training_shape=shape,
                  tokens_per_s=total_tokens/result['block_wall_s'],total_measured_tokens=total_tokens,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  optimizer_steps=state['opt_steps'],expected_optimizer_steps=total,
                  effective_batch_sequences=b*accum,losses=[float(x.item()) for x in state['losses']],
                  final_grad_scale=float(scaler.get_scale()),
                  budget_scope='whole-step allocated bytes, unlike EXTRA-memory operator selection')
    result['training_budget_bytes']=int(torch.cuda.get_device_properties(0).total_memory*cfg['training_budget_fraction'])
    result['within_training_budget']=result['peak_allocated_bytes']<=result['training_budget_bytes']
    hook.remove();del opt,scaler,model,bank;gc.collect();torch.cuda.empty_cache()
    return result
