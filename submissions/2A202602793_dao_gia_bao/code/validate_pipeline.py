"""EDA và kiểm tra trước thực nghiệm; chỉ xem ảnh train, test còn niêm phong."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from dataset import load_split, check_split, build_transforms, denormalize, DeepWeedsDataset, set_seed
from losses import mix_batch
from model import build_model, SCREENING_MODELS
from paths import SUB, REPO, DATA
from train import Config, build_optimizer, build_scheduler, train_one_epoch, resolve_device, synchronize, write_json, versions

EVIDENCE=SUB/'evidence'
IMAGES=DATA/'images'
LABELS=DATA/'labels'


def eda(train,val,test,split):
    names=split['class_names']
    counts=pd.DataFrame({'Label':range(9),'Lớp':names,**split['per_class']})
    ref=pd.read_csv(LABELS/'labels.csv')
    counts['labels.csv']=ref.Label.value_counts().reindex(range(9)).to_numpy()
    counts['Chênh split/nguồn']=counts[['train','val','test']].sum(1)-counts['labels.csv']
    counts.to_csv(EVIDENCE/'class_counts.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(9)
    for offset,s in zip([-.25,0,.25],['train','val','test']):
        bars=ax.bar(x+offset,counts[s],width=.25,label=s)
        ax.bar_label(bars,fontsize=7,padding=2,rotation=90)
    ax.set_xticks(x,names,rotation=30,ha='right'); ax.set_ylabel('Số ảnh'); ax.legend()
    ax.set_title('DeepWeeds fold 0 — phân bố lớp từ CSV nguyên bản')
    ax.set_ylim(0,counts.train.max()*1.18)
    fig.tight_layout(); fig.savefig(EVIDENCE/'class_distribution.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(9,3,figsize=(10,24))
    sample_rows=[]
    for label in range(9):
        rows=train[train.Label==label].iloc[:3]
        for ax,row in zip(axes[label],rows.itertuples()):
            with Image.open(IMAGES/row.Filename) as im:
                ax.imshow(im.convert('RGB'))
            ax.set_title(f'{names[label]}\n{row.Filename}',fontsize=8); ax.axis('off')
            sample_rows.append({'Filename':row.Filename,'Label':int(row.Label),'split':'train'})
    fig.suptitle('Ba ảnh train mỗi lớp — mẫu cố định',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(EVIDENCE/'class_samples.png',dpi=120); plt.close(fig)
    pd.DataFrame(sample_rows).to_csv(EVIDENCE/'class_samples.csv',index=False)
    sizes,modes=Counter(),Counter()
    for filename in train.Filename:
        with Image.open(IMAGES/filename) as im:
            sizes[str(im.size)]+=1; modes[im.mode]+=1
    ds=DeepWeedsDataset(train.iloc[:12],IMAGES,build_transforms(True,128))
    set_seed(0)
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    tensors,targets=[],[]
    for i,ax in enumerate(axes.flat):
        image,label,filename=ds[i]
        assert torch.isfinite(image).all()
        ax.imshow(denormalize(image).permute(1,2,0))
        ax.set_title(f'{names[label]}\n{filename}',fontsize=8); ax.axis('off')
        tensors.append(image); targets.append(label)
    fig.suptitle('RandomResizedCrop + lật ngang, đã giải chuẩn hóa')
    fig.tight_layout(); fig.savefig(EVIDENCE/'augmentation_samples.png',dpi=140); plt.close(fig)
    x,y=torch.stack(tensors[:8]),torch.tensor(targets[:8])
    mixed_records=[]
    fig,axes=plt.subplots(2,4,figsize=(12,7))
    for row,mode in enumerate(['mixup','cutmix']):
        mixed,(ya,yb,lam)=mix_batch(x,y,mode=mode)
        for i,ax in enumerate(axes[row]):
            ax.imshow(denormalize(mixed[i]).permute(1,2,0))
            ax.set_title(f'{mode}, λ={lam:.3f}\n{names[int(ya[i])]} / {names[int(yb[i])]}',fontsize=8)
            ax.axis('off')
        mixed_records.append({'mode':mode,'lambda':lam})
    fig.tight_layout(); fig.savefig(EVIDENCE/'mixed_samples.png',dpi=140); plt.close(fig)
    total=counts[['train','val','test']].sum(1).to_numpy()
    return {'image_dimensions_train':dict(sizes),'image_modes_train':dict(modes),
            'negative_fraction':float(total[8]/total.sum()),'largest_smallest_ratio':float(total.max()/total.min()),
            'samples_per_class':3,'visualized_split':'train','mixed_visualizations':mixed_records}


def sanity(train,device):
    set_seed(0)
    picked=train.groupby('Label',sort=True).head(1).sort_values('Label')
    ds=DeepWeedsDataset(picked,IMAGES,build_transforms(False,128))
    entries=[ds[i] for i in range(9)]
    x=torch.stack([r[0] for r in entries]).to(device)
    y=torch.tensor([r[1] for r in entries],device=device)
    filenames=[r[2] for r in entries]
    m=build_model('mobilenetv3_small_100.lamb_in1k',True,init='frozen',img_size=128).to(device)
    m.eval()
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        initial=float(criterion(m(x),y))
    if not 1.5<initial<3.:
        raise RuntimeError(f'Loss ban đầu ngoài khoảng kiểm tra: {initial}')
    cfg=Config(exp_id='SANITY_one_batch',init='frozen',epochs=400,lr_head=.02,weight_decay=0,
               warmup_epochs=0,scheduler='none',amp=False,img_size=128,batch_size=9)
    opt=build_optimizer(m,cfg); scheduler=build_scheduler(opt,cfg,1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    history=[]
    for step in range(1,401):
        train_one_epoch(m,[(x,y,filenames)],criterion,opt,scheduler,scaler,cfg,device)
        m.eval()
        with torch.no_grad():
            z=m(x); loss=float(criterion(z,y)); acc=float((z.argmax(1)==y).float().mean())
        history.append({'step':step,'loss':loss,'top1':acc})
        if acc==1. and loss<.05:
            break
    if history[-1]['top1']!=1. or history[-1]['loss']>=.05:
        raise RuntimeError(f'Không ghi nhớ được batch: {history[-1]}')
    pd.DataFrame(history).to_csv(EVIDENCE/'one_batch_history.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in history],[r['loss'] for r in history]); axes[0].set_ylabel('CE')
    axes[1].plot([r['step'] for r in history],[r['top1'] for r in history]); axes[1].set_ylabel('Top-1 train')
    for ax in axes: ax.set_xlabel('Bước tối ưu'); ax.grid(alpha=.2)
    fig.suptitle('SANITY_one_batch — 9 ảnh train cố định, backbone đóng băng')
    fig.tight_layout(); fig.savefig(EVIDENCE/'one_batch_overfit.png',dpi=150); plt.close(fig)
    return {'initial_ce':initial,'reference_ln9':float(np.log(9)),
            'overfit':history[-1],'filenames':filenames,'split':'train',
            'configuration':asdict(cfg),'versions':versions(device)}


def budget_probe(device):
    rows=[]
    for name in SCREENING_MODELS:
        for size in [224,128]:
            set_seed(0)
            m=build_model(name,pretrained=False,img_size=size).to(device).train()
            cfg=Config(backbone=name,batch_size=32,img_size=size,amp=False)
            opt=build_optimizer(m,cfg)
            x=torch.randn(32,3,size,size,device=device); y=torch.arange(32,device=device)%9
            def step():
                opt.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(m(x),y).backward(); opt.step()
            for _ in range(3): step()
            times=[]
            for _ in range(5):
                synchronize(device); start=time.perf_counter(); step(); synchronize(device)
                times.append(time.perf_counter()-start)
            row={'backbone':name,'resolution':size,'batch_size':32,'dtype':'fp32',
                 'train_batch_mean_seconds':float(np.mean(times)),
                 'estimated_train_epoch_seconds_no_io':float(np.mean(times)*(10501//32)),
                 'warmups':3,'timed_batches':5,'purpose':'Ước lượng tính toán, không phải đo độ trễ suy luận hay điểm model'}
            rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
            del m,opt,x,y
            if device.type=='mps': torch.mps.empty_cache()
    pd.DataFrame(rows).to_csv(EVIDENCE/'compute_probe.csv',index=False)
    return rows


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); set_seed(0)
    device=resolve_device()
    train,val,test=load_split(LABELS)
    split=check_split(train,val,test,IMAGES)
    result={'split':split,'eda':eda(train,val,test,split),'sanity':sanity(train,device),
            'compute_probe':budget_probe(device)}
    write_json(EVIDENCE/'stage5.json',result)
    print('EDA và kiểm tra pipeline đạt',flush=True)


if __name__=='__main__': main()
