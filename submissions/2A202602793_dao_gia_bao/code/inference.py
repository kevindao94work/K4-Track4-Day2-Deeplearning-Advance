"""Suy luận xác định; nhiệt độ khớp trên validation, không chọn bằng test."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


def _logits(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=9 or len(a)==0 or not np.isfinite(a).all():
        raise ValueError('Cần logits hữu hạn [N,9] không rỗng')
    return a


def apply_temperature(logits,T:float):
    if not np.isfinite(T) or T<=0:raise ValueError('Nhiệt độ phải hữu hạn và dương')
    a=_logits(logits)/float(T)
    return np.exp(a-logsumexp(a,axis=1,keepdims=True))


def fit_temperature(val_logits,val_labels)->float:
    a=_logits(val_logits);y=np.asarray(val_labels)
    if y.shape!=(len(a),) or not np.isin(y,np.arange(9)).all():raise ValueError('Nhãn val không hợp lệ')
    y=y.astype(np.int64)
    def objective(log_t):
        z=a/np.exp(log_t)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    result=minimize_scalar(objective,bounds=(-5.,5.),method='bounded',options={'xatol':1e-9})
    if not result.success:raise RuntimeError('Tối ưu nhiệt độ không hội tụ')
    # T=1 thuộc tập ứng viên để không tăng NLL do sai số tối ưu.
    return float(np.exp(result.x)) if result.fun<objective(0.) else 1.


def view_identity(x):return x


def view_hflip(x):
    if x.ndim!=4:raise ValueError('Ảnh phải [N,C,H,W]')
    return torch.flip(x,dims=(-1,))


def views_multicrop(x,crop:int):
    if x.ndim!=4 or crop<=0 or crop>min(x.shape[-2:]):raise ValueError('Crop ngoài ảnh')
    h,w=x.shape[-2:]
    offsets=[(0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2)]
    return [x[...,a:a+crop,b:b+crop] for a,b in offsets]


def views_multiscale(x,sizes):
    if x.ndim!=4 or not sizes or any(int(s)!=s or s<=0 for s in sizes):raise ValueError('Kích thước scale sai')
    # Bilinear không antialias chạy native MPS 2.6; bicubic antialias chưa hỗ trợ.
    return [F.interpolate(x,size=(int(s),int(s)),mode='bilinear',align_corners=False) for s in sizes]


def aggregate_views(logits_per_view,space:str='prob'):
    arrays=[_logits(a) for a in logits_per_view]
    if not arrays or any(a.shape!=arrays[0].shape for a in arrays):raise ValueError('View khác shape')
    if space=='prob':return ensemble_probs([apply_temperature(a,1.) for a in arrays])
    if space=='logit':return apply_temperature(np.mean(arrays,axis=0),1.)
    raise ValueError('Gộp phải prob hoặc logit')


def ensemble_probs(list_of_probs):
    arrays=[_logits(p) for p in list_of_probs]
    if not arrays or any(p.shape!=arrays[0].shape for p in arrays):raise ValueError('Ensemble khác shape')
    if any((p<0).any() or (p>1+1e-6).any() or not np.allclose(p.sum(1),1,atol=1e-6) for p in arrays):
        raise ValueError('Ensemble cần xác suất đã chuẩn hóa, cùng thứ tự ảnh')
    p=np.mean(arrays,axis=0)
    return p/p.sum(1,keepdims=True)


def predict_logits(model,loader,device,view=None):
    model.eval();device=torch.device(device);names=[];targets=[];outputs=[]
    with torch.inference_mode():
        for x,y,files in loader:
            x=x.to(device)
            if view is not None:x=view(x)
            z=model(x)
            names.extend(map(str,files));targets.append(y.cpu().numpy());outputs.append(z.float().cpu().numpy())
    if not outputs:raise ValueError('Loader rỗng')
    if len(set(names))!=len(names):raise ValueError('Tên ảnh trùng khi suy luận')
    logits=np.concatenate(outputs);_logits(logits)
    return names,np.concatenate(targets).astype(np.int64),logits


def fuse_conv_bn(model):
    """Gộp bản sao eval; chỉ cặp Sequential hoặc tên convN/bnN có cấu trúc rõ."""
    fused=copy.deepcopy(model).eval();count=0
    def visit(parent):
        nonlocal count
        children=list(parent.named_children())
        for _,child in children:visit(child)
        for (cn,conv),(bn,batchnorm) in zip(children,children[1:]):
            known=isinstance(parent,nn.Sequential) or (cn.startswith('conv') and bn=='bn'+cn[4:])
            # BNAct có thêm activation: không được thay toàn bộ bằng Identity.
            if known and isinstance(conv,nn.Conv2d) and type(batchnorm) is nn.BatchNorm2d:
                setattr(parent,cn,torch.nn.utils.fusion.fuse_conv_bn_eval(conv,batchnorm))
                setattr(parent,bn,nn.Identity());count+=1
    visit(fused);fused.lab_fused_pairs=count
    return fused


@dataclass(frozen=True)
class Method:
    exp_id:str='I00'
    name:str='Một view'
    policy:str='identity'
    resolution:int=128
    input_size:int=128
    aggregation:str='logit'
    sizes:tuple=(96,128,160)

    @property
    def k(self):return {'identity':1,'hflip':2,'fivecrop':5,'multiscale':len(self.sizes)}[self.policy]

    def views(self,x):
        if self.policy=='identity':return [x]
        if self.policy=='hflip':return [x,view_hflip(x)]
        if self.policy=='fivecrop':return views_multicrop(x,self.resolution)
        if self.policy=='multiscale':return views_multiscale(x,self.sizes)
        raise ValueError('Policy suy luận chưa hỗ trợ')


def method_logits(model,x,method):
    views=[model(v) for v in method.views(x)]
    if method.aggregation=='logit':return torch.stack(views).mean(0)
    if method.aggregation=='prob':
        return torch.stack([z.softmax(-1) for z in views]).mean(0).clamp_min(torch.finfo(views[0].dtype).tiny).log()
    raise ValueError('Không gian gộp sai')


def predict_method(model,loader,device,method):
    """Một vòng qua loader, K forward cố định mỗi batch; trả logit tương đương."""
    model.eval();device=torch.device(device);names=[];ys=[];zs=[]
    with torch.inference_mode():
        for x,y,files in loader:
            z=method_logits(model,x.to(device),method)
            names.extend(map(str,files));ys.append(y.numpy());zs.append(z.float().cpu().numpy())
    if not zs or len(set(names))!=len(names):raise ValueError('Loader rỗng/tên trùng')
    logits=np.concatenate(zs);_logits(logits)
    return names,np.concatenate(ys).astype(np.int64),logits


def export_frozen_test(cfg):
    """Giao diện engine; ledger và cổng đủ ba seed nằm trong test_once."""
    from test_once import export_frozen_test as export
    return export(cfg)
