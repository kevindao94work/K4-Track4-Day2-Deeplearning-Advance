"""Độ trễ forward thực đo; 10 warmup, >=50 mẫu, đồng bộ CUDA/MPS."""
from __future__ import annotations
from contextlib import nullcontext
import copy
import time
import numpy as np
import torch
from train import synchronize,resolve_device,hardware_name


def bench(fn,warmup:int=10,iters:int=100,sync=None)->dict:
    if warmup<10 or iters<50:raise ValueError('Cần >=10 warmup và >=50 lần đo')
    sync=sync or (lambda:None);times=[]
    with torch.inference_mode():
        for _ in range(warmup):fn()
        sync()
        for _ in range(iters):
            sync();start=time.perf_counter();fn();sync()
            times.append((time.perf_counter()-start)*1000)
    q=np.percentile(times,[50,95,99])
    return dict(p50=float(q[0]),p95=float(q[1]),p99=float(q[2]),mean=float(np.mean(times)),n=iters,warmups=warmup,samples_ms=times)


def latency_report(model,batch_size:int,img_size:int,dtype:str='fp32',device:str='cuda',warmup:int=10,iters:int=100)->dict:
    device=resolve_device(device)
    if dtype not in ('fp32','amp','fp16') or min(batch_size,img_size)<=0:raise ValueError('Điều kiện benchmark sai')
    if dtype=='amp' and device.type!='cuda':raise ValueError('AMP benchmark này chỉ dùng CUDA; MPS dùng FP32/FP16 tường minh')
    m=copy.deepcopy(model).to(device).eval()
    m=m.half() if dtype=='fp16' else m.float()
    x=torch.randn(batch_size,3,img_size,img_size,device=device,dtype=torch.float16 if dtype=='fp16' else torch.float32)
    def forward():
        with torch.autocast('cuda',dtype=torch.float16) if dtype=='amp' else nullcontext():return m(x)
    r=bench(forward,warmup,iters,sync=lambda:synchronize(device))
    r.update(gpu=hardware_name(device),dtype=dtype,batch=batch_size,img_size=img_size,
             images_per_s=batch_size/(r['p50']/1000),torch=torch.__version__,device=str(device),
             bn_fusion=bool(getattr(m,'lab_fused_pairs',0)),preprocessing_included=False)
    return r


def tta_latency(model,k_views:int,**kw)->dict:
    if k_views<1:raise ValueError('K phải dương')
    device=resolve_device(kw.pop('device','cuda'));batch=kw.pop('batch_size',1);size=kw.pop('img_size',224)
    dtype=kw.pop('dtype','fp32');warmup=kw.pop('warmup',10);iters=kw.pop('iters',100)
    if kw or dtype!='fp32':raise ValueError('TTA generic dùng FP32; policy thực đo riêng qua method_latency')
    m=copy.deepcopy(model).to(device).float().eval();x=torch.randn(batch,3,size,size,device=device)
    def forward():return torch.stack([m(x).softmax(-1) for _ in range(k_views)]).mean(0)
    r=bench(forward,warmup,iters,lambda:synchronize(device))
    r.update(K=k_views,gpu=hardware_name(device),dtype=dtype,batch=batch,img_size=size,
             images_per_s=batch/(r['p50']/1000),torch=torch.__version__,preprocessing_included=False,
             note='K forward thực đo; biến đổi view cụ thể đo bằng method_latency')
    return r


def method_latency(model,method,device,batch=1,temperature=1.,iters=50):
    from inference import method_logits
    device=resolve_device(str(device));model.eval()
    x=torch.randn(batch,3,method.input_size,method.input_size,device=device)
    def forward():return (method_logits(model,x,method)/temperature).softmax(-1)
    r=bench(forward,10,iters,lambda:synchronize(device))
    r.update(gpu=hardware_name(device),dtype='fp32',batch=batch,img_size=method.resolution,
             input_size=method.input_size,K=method.k,bn_fusion=False,torch=torch.__version__,
             images_per_s=batch/(r['p50']/1000),preprocessing_included=False,
             condition='Tensor đã chuẩn hóa trên GPU; gồm biến đổi view, forward, gộp và temperature/softmax; không đọc/resize/normalize ảnh CPU')
    return r
