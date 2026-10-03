"""Smoke giai đoạn 4 trên ảnh thật; không dùng kết quả nhỏ để chọn model."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch
from torch import nn

from dataset import load_split, make_loader, build_transforms, set_seed
from model import build_model, classifier
from train import Config, run, evaluate, probabilities, resolve_device
import train as engine


def main():
    sub=Path(__file__).resolve().parents[1]
    repo=sub.parents[1]
    torch.set_num_threads(4)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='lab_smoke_') as tmp:
        for mode in ['finetune','frozen']:
            cfg=Config(exp_id='SMOKE_'+mode,seed=0,backbone='mobilenetv3_small_100.lamb_in1k',
                       pretrained=False,init=mode,img_size=128,epochs=1,batch_size=8,warmup_epochs=0,
                       num_workers=0,smoke_train=32,smoke_val=18,amp=False,
                       ema_decay=.9 if mode=='finetune' else None,
                       images_dir=str(repo/'data/images'),labels_dir=str(repo/'data/labels'),
                       out_dir=str(Path(tmp)/'logs'),pred_dir=str(Path(tmp)/'predictions'),
                       curves_dir=str(sub/'evidence/smoke_curves'))
            set_seed(cfg.seed)
            before=build_model(cfg.backbone,False,init=mode,img_size=128)
            initial={k:v.clone() for k,v in before.state_dict().items()}
            head_names={n for n,p in before.named_parameters() if id(p) in {id(t) for t in classifier(before).parameters()}}
            summary=run(cfg)
            out=Path(summary['best_checkpoint']).parent
            saved=torch.load(summary['best_checkpoint'],map_location='cpu',weights_only=False)
            assert saved['epoch']==summary['best_epoch']==1
            assert any(not torch.equal(initial[n],saved['model'][n]) for n in head_names)
            if mode=='frozen':
                for k,v in initial.items():
                    if k not in head_names:
                        assert torch.equal(v,saved['model'][k]), k
            fresh=build_model(cfg.backbone,False,init=mode,img_size=128)
            fresh.load_state_dict(saved['model']); device=resolve_device(cfg.device); fresh.to(device)
            _,val,_=load_split(cfg.labels_dir)
            loader=make_loader(val.iloc[:18],cfg.images_dir,build_transforms(False,128),8,False,num_workers=0)
            names,y,z,loss=evaluate(fresh,loader,nn.CrossEntropyLoss(),device)
            old=np.load(out/'val_outputs.npz')
            np.testing.assert_allclose(z,old['logits'],atol=1e-5,rtol=1e-5)
            assert Path(summary['curve']).is_file() and (out/'history.csv').is_file() and (out/'config.json').is_file()
            evidence_dir=sub/'evidence/stage4_runs'/mode
            evidence_dir.mkdir(parents=True,exist_ok=True)
            for filename in ['config.json','history.csv']:
                shutil.copyfile(out/filename,evidence_dir/filename)
            rows.append({'mode':mode,'best_epoch':summary['best_epoch'],'val_loss':loss,
                         'macro_f1':summary['val']['macro_f1'],'weights_updated':True,
                         'frozen_unchanged':mode=='frozen','checkpoint_reloaded':True,
                         'fresh_eval_max_logit_error':float(np.max(np.abs(z-old['logits']))),
                         'curve':str(Path(summary['curve']).relative_to(sub))})
        base=replace(cfg,exp_id='SMOKE_resume_reference',init='finetune',epochs=2,device='cpu',
                     smoke_train=16,smoke_val=9,ema_decay=None)
        reference=run(base)
        interrupted=replace(base,exp_id='SMOKE_resume_interrupted')
        original=engine.train_one_epoch
        calls=0
        def stop_after_epoch(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:
                raise RuntimeError('Gián đoạn chủ động để kiểm tra resume')
            return original(*args,**kwargs)
        engine.train_one_epoch=stop_after_epoch
        try:
            run(interrupted)
        except RuntimeError as exc:
            assert 'Gián đoạn chủ động' in str(exc)
        finally:
            engine.train_one_epoch=original
        resumed=run(interrupted)
        a=torch.load(Path(reference['best_checkpoint']).parent/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(Path(resumed['best_checkpoint']).parent/'latest.pt',map_location='cpu',weights_only=False)
        for name in a['model']:
            torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)
        assert resumed['epochs_completed']==2
        rows.append({'mode':'resume_cpu','epochs_completed':2,'weights_match_uninterrupted_exactly':True})
        evidence_dir=sub/'evidence/stage4_runs/resume_cpu'
        evidence_dir.mkdir(parents=True,exist_ok=True)
        for filename in ['config.json','history.csv']:
            shutil.copyfile(Path(resumed['best_checkpoint']).parent/filename,evidence_dir/filename)
    (sub/'evidence/stage4.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
