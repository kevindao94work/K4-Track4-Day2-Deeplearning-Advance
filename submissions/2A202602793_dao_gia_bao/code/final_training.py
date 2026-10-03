"""Huấn luyện chung kết ba seed, validation trước; không suy luận test."""
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from paths import SUB,DATA
from freeze_final import frozen_path,config_for,assert_integrity
from train import run,run_dir,write_json,frozen_recipe
from inference import Method,predict_method,fit_temperature,apply_temperature
from inference_lab import load_checkpoint,eval_loader
from benchmark import method_latency
from eval import save_predictions,read_pred,check_against_csv,compute_metrics,mean_std


def final_val(cfg,summary,frozen):
    folder=run_dir(cfg);record_path=folder/'final_val.json'
    freeze_sha=hashlib.sha256(frozen_path().read_bytes()).hexdigest()
    if record_path.is_file():
        record=json.loads(record_path.read_text())
        if record['frozen_sha256']!=freeze_sha or record['checkpoint_sha256']!=summary['checkpoint_sha256']:
            raise ValueError('Val cache khác cấu hình/checkpoint chốt')
        p=read_pred(SUB/'predictions'/f'{cfg.exp_id}_seed{cfg.seed}_val.csv')
        check_against_csv(p,DATA/'labels/val_subset0.csv','val')
        if hashlib.sha256(Path(p.path).read_bytes()).hexdigest()!=record['prediction_sha256']:
            raise ValueError('CSV val cache đã bị sửa')
        return record
    # Giữ riêng dự đoán val dùng chọn checkpoint ở độ phân giải train.
    source=SUB/'predictions'/f'{cfg.exp_id}_seed{cfg.seed}_val.csv'
    training_backup=SUB/'predictions'/f'{cfg.exp_id}_train_seed{cfg.seed}_val.csv'
    if not training_backup.is_file():shutil.copyfile(source,training_backup)
    spec=frozen['configs'][cfg.exp_id];method=Method(**spec['inference'])
    if method.policy=='identity' and method.input_size==cfg.img_size:
        data=np.load(folder/'val_outputs.npz',allow_pickle=False)
        names=data['filenames'].tolist();y=data['y_true'];z=data['logits']
    else:
        model,device,_=load_checkpoint(summary)
        names,y,z=predict_method(model,eval_loader(cfg,method),device,method)
    temperature=fit_temperature(z,y) if spec['temperature_policy']!='none' else 1.
    raw=apply_temperature(z,1.);cal=apply_temperature(z,temperature)
    if not np.array_equal(raw.argmax(1),cal.argmax(1)):raise ValueError('TS đổi argmax')
    raw_path=SUB/'predictions'/f'{cfg.exp_id}_uncal_seed{cfg.seed}_val.csv'
    save_predictions(raw_path,names,y,raw);save_predictions(source,names,y,cal)
    raw_pred=read_pred(raw_path);pred=read_pred(source)
    check_against_csv(pred,DATA/'labels/val_subset0.csv','val')
    raw_metrics=compute_metrics(raw_pred.y_true,raw_pred.y_pred,raw_pred.probs)
    metrics=compute_metrics(pred.y_true,pred.y_pred,pred.probs)
    np.savez(folder/'final_val_outputs.npz',filenames=np.asarray(names),y_true=y,logits=z)
    record={'exp_id':cfg.exp_id,'seed':cfg.seed,'method':asdict(method),
        'temperature':temperature,'temperature_fit_split':'val' if spec['temperature_policy']!='none' else None,
        'temperature_policy':spec['temperature_policy'],'val':metrics,'uncal_val':raw_metrics,
        'frozen_sha256':freeze_sha,'checkpoint_sha256':summary['checkpoint_sha256'],
        'prediction_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'test_evaluated':False}
    write_json(record_path,record)
    return record


def final_training():
    frozen=json.loads(frozen_path().read_text());assert_integrity(frozen)
    gate=json.loads((SUB/'evidence/stage9.json').read_text())
    if not gate['passed'] or gate['frozen_sha256']!=hashlib.sha256(frozen_path().read_bytes()).hexdigest():
        raise ValueError('Dry run cấu hình chốt chưa đạt')
    rows=[];records=[];summaries={}
    for eid in ['F01','T00']:
        for seed in frozen['seeds']:
            assert_integrity(frozen);cfg=config_for(eid,seed)
            summary=run(cfg)
            if summary['epochs_completed']!=cfg.epochs or summary['val']['n']!=3501 or summary['test_evaluated']:
                raise ValueError('Chung kết phải đủ train/val và chưa test')
            if summary['model']['pretrained_source']['sha256']!=frozen['configs'][eid]['pretrained_source']['sha256']:
                raise ValueError('Trọng số tiền huấn luyện khác nguồn chốt')
            if frozen_recipe(cfg)!=frozen['configs'][eid]['recipe']:raise ValueError('Công thức đã đổi')
            record=final_val(cfg,summary,frozen);records.append(record);summaries[(eid,seed)]=summary
            rows.append({'exp_id':eid,'seed':seed,'macro_f1_val':record['val']['macro_f1'],
                'top1_val':record['val']['top1'],'ece_val':record['val']['ece'],'nll_val':record['val']['nll'],
                'temperature':record['temperature'],'best_epoch':summary['best_epoch'],
                'macro_f1_val_checkpoint_train_resolution':summary['val']['macro_f1'],
                'checkpoint_sha256':summary['checkpoint_sha256'],'test_evaluated':False})
            pd.DataFrame(rows).to_csv(SUB/'tables/FinalValidation.csv',index=False)
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    groups={}
    for eid in ['F01','T00']:
        selected=[r for r in rows if r['exp_id']==eid]
        groups[eid]={k:dict(zip(['mean','std'],mean_std([r[k] for r in selected]))) for k in ['macro_f1_val','top1_val','ece_val','nll_val']}
    # Đo lại trên đúng checkpoint seed0 chung kết, policy cố định.
    latency_rows=[]
    for eid in ['F01','T00']:
        summary=summaries[(eid,0)];model,device,cfg=load_checkpoint(summary)
        record=next(r for r in records if r['exp_id']==eid and r['seed']==0)
        method=Method(**frozen['configs'][eid]['inference'])
        for batch in [1,32]:
            latency=method_latency(model,method,device,batch,record['temperature'],50)
            latency['configuration']=eid+'_seed0';latency['method']='Policy chung kết đã chốt'
            latency['checkpoint_sha256']=summary['checkpoint_sha256']
            write_json(run_dir(cfg)/f'batch{batch}_latency.json',latency)
            latency_rows.append({k:v for k,v in latency.items() if k!='samples_ms'})
    screening=pd.read_csv(SUB/'tables/Latency.csv')
    screening=screening[~screening.configuration.astype(str).isin(['F01_seed0','T00_seed0'])]
    pd.concat([screening,pd.DataFrame(latency_rows)],ignore_index=True).to_csv(SUB/'tables/Latency.csv',index=False)
    gate={'passed':True,'seeds':frozen['seeds'],'groups':groups,'ddof':1,'runs':rows,
          'frozen_sha256':hashlib.sha256(frozen_path().read_bytes()).hexdigest(),'test_evaluated':False}
    write_json(SUB/'evidence/stage10.json',gate)
    lines=['# Chung kết — validation trước test','',
           '| ID | Seed | Macro-F1 val | Top-1 val | ECE val | T từ val |',
           '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['exp_id']} | {r['seed']} | {r['macro_f1_val']:.4f} | {r['top1_val']:.4f} | {r['ece_val']:.4f} | {r['temperature']:.5f} |" for r in rows]
    lines += ['',json.dumps(groups,ensure_ascii=False,indent=2),'',
              'Std là sample std ddof=1. Mỗi seed huấn luyện đủ 10 epoch bằng recipe chốt; checkpoint chọn trên val128, sau đó áp dụng inference cố định. Không chọn lại checkpoint/công thức bằng kết quả chung kết. Test chưa chạy.']
    (SUB/'evidence/final_validation.md').write_text('\n'.join(lines)+'\n')
    return gate


if __name__=='__main__':final_training()
