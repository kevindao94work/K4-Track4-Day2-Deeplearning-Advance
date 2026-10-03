"""Điều phối thực nghiệm theo giai đoạn; tất cả huấn luyện gọi train.run."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from paths import CODE, SUB, DATA, REPO
from model import build_model, SCREENING_MODELS
from train import Config, run, resolve_device, synchronize, hardware_name, write_json
from eval import read_pred, check_against_csv, compute_metrics

B_IDS=['B01_resnet18','B02_convnext_atto','B03_deit_tiny','B04_efficientnet_b0','B05_mobilenetv3_small']


def audit_backbones():
    """Đối chiếu bằng chứng thật; không thực hiện thêm forward của mô hình."""
    table=pd.read_csv(SUB/'tables/Backbones.csv')
    if table.exp_id.tolist()!=B_IDS or len(table)!=5:
        raise ValueError('Thiếu một trong năm backbone bắt buộc')
    reference=None; audited=[]
    for exp_id in B_IDS:
        folder=SUB/'logs'/exp_id/'seed0'
        record=json.loads((folder/'config.json').read_text())
        cfg=record['config']
        controlled={k:v for k,v in cfg.items() if k not in ('exp_id','backbone')}
        if reference is None: reference=controlled
        if controlled!=reference: raise ValueError(f'{exp_id}: cấu hình nền khác nhau')
        history=pd.read_csv(folder/'history.csv')
        if history.epoch.tolist()!=list(range(1,cfg['epochs']+1)):
            raise ValueError(f'{exp_id}: lịch sử thiếu epoch')
        summary=json.loads((folder/'summary.json').read_text())
        prediction=read_pred(SUB/'predictions'/f'{exp_id}_seed0_val.csv')
        check_against_csv(prediction,DATA/'labels/val_subset0.csv','val')
        metrics=compute_metrics(prediction.y_true,prediction.y_pred,prediction.probs)
        row=table.loc[table.exp_id==exp_id].iloc[0]
        for key,column in [('macro_f1','macro_f1_val'),('top1','top1_val')]:
            if not np.isclose(metrics[key],summary['val'][key],atol=1e-12,rtol=0):
                raise ValueError(f'{exp_id}: summary không khớp evaluator')
            if not np.isclose(metrics[key],row[column],atol=1e-12,rtol=0):
                raise ValueError(f'{exp_id}: bảng không khớp evaluator')
        curve=Path(summary['curve'])
        if not curve.is_file(): raise ValueError(f'{exp_id}: thiếu biểu đồ')
        if summary['test_evaluated'] or cfg['save_test_predictions']:
            raise ValueError('Test phải tiếp tục niêm phong trong sàng lọc')
        audited.append({'exp_id':exp_id,'epochs':len(history),'val_n':len(prediction.filenames),
                        'macro_f1_val':metrics['macro_f1'],'curve':str(curve),
                        'test_evaluated':False})
    gate={'passed':True,'controlled_fields':reference,'runs':audited,
          'evaluator_source':'eval.py nguyên bản; đối chiếu CSV, không chạy test'}
    write_json(SUB/'evidence/stage6.json',gate)
    return gate


def baseline_config(exp_id='T00',seed=0):
    source=CODE.parent/'configs/screening_recipe.json'
    record=json.loads(source.read_text())
    cfg=Config(**record['recipe'],exp_id=exp_id,seed=seed,
               images_dir=str(DATA/'images'),labels_dir=str(DATA/'labels'),
               out_dir=str(SUB/'logs'),pred_dir=str(SUB/'predictions'),curves_dir=str(SUB/'curves'))
    target=SUB/'configs/screening_recipe.json'
    if source.resolve()!=target.resolve(): write_json(target,record)
    return cfg


def pilot_latency(summary):
    cfg=Config(**summary['config']);device=resolve_device(cfg.device)
    m=build_model(cfg.backbone,False,init=cfg.init,img_size=cfg.img_size).to(device).eval()
    checkpoint=torch.load(summary['best_checkpoint'],map_location='cpu',weights_only=False)
    m.load_state_dict(checkpoint['model'])
    x=torch.randn(1,3,cfg.img_size,cfg.img_size,device=device)
    times=[]
    with torch.inference_mode():
        for _ in range(10): m(x)
        for _ in range(5):
            synchronize(device);start=time.perf_counter();m(x);synchronize(device)
            times.append((time.perf_counter()-start)*1000)
    return {'p50_ms':float(np.median(times)),'warmups':10,'iterations':5,'batch':1,
            'gpu':hardware_name(device),'dtype':'fp32','resolution':cfg.img_size,
            'preprocessing_included':False,'purpose':'Độ trễ sơ bộ; đo chính thức ở giai đoạn suy luận'}


def backbones():
    gate=SUB/'evidence/stage5.json'
    if not gate.is_file(): raise ValueError('Phải chạy validate_pipeline trước sàng lọc')
    evidence=json.loads(gate.read_text())
    if evidence['split']['union']!=17509 or evidence['sanity']['overfit']['loss']>=.05:
        raise ValueError('Cổng pipeline chưa đạt')
    baseline=baseline_config()
    rows=[]; summaries=[]
    for exp_id,name in zip(B_IDS,SCREENING_MODELS):
        cfg=replace(baseline,exp_id=exp_id,backbone=name)
        summary=run(cfg)
        if summary['epochs_completed']!=cfg.epochs or summary['val']['n']!=3501:
            raise ValueError('Không đủ epoch hoặc toàn bộ val')
        pilot=pilot_latency(summary)
        write_json(Path(summary['best_checkpoint']).parent/'pilot_latency.json',pilot)
        summaries.append(summary)
        row={'exp_id':exp_id,'backbone':name,'pretrained_tag':summary['model']['pretrained_tag'],
             'pretrained_revision':summary['model']['pretrained_source']['revision'],
             'parameters_m':summary['model']['parameters_m'],'gmac':summary['model']['gmac'],
             'resolution':cfg.img_size,'epochs':cfg.epochs,'seed':cfg.seed,
             'macro_f1_val':summary['val']['macro_f1'],'top1_val':summary['val']['top1'],
             'training_seconds_per_epoch':summary['training_seconds_per_epoch'],
             'seconds_per_epoch_train_val':summary['seconds_per_epoch'],
             'batch1_latency_ms':pilot['p50_ms'],'best_epoch':summary['best_epoch'],
             'notes':'Một seed sàng lọc; latency sơ bộ 10 warmup/5 lần, không tính tiền xử lý'}
        rows.append(row)
        (SUB/'tables').mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(SUB/'tables/Backbones.csv',index=False)
    ordered=sorted(rows,key=lambda r:(-r['macro_f1_val'],r['batch1_latency_ms']))
    chosen=next(s for s in summaries if s['exp_id']==ordered[0]['exp_id'])
    choice={'source_exp_id':chosen['exp_id'],'config':chosen['config'],
            'macro_f1_val':chosen['val']['macro_f1'],'top1_val':chosen['val']['top1'],
            'selection_split':'val','screening_seed':0,
            'reason':'Chọn macro-F1 val cao nhất; khi hòa chọn latency thấp hơn. Chỉ một seed, thứ hạng còn sơ bộ.',
            'all_results':ordered}
    write_json(SUB/'configs/backbone_choice.json',choice)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for row in rows:
        ax.scatter(row['batch1_latency_ms'],row['macro_f1_val'],s=65)
        ax.annotate(row['exp_id'],(row['batch1_latency_ms'],row['macro_f1_val']),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Độ trễ sơ bộ p50 batch 1 (ms, 5 lần đo)');ax.set_ylabel('Macro-F1 val')
    ax.set_title('Sàng lọc backbone — T00, cùng seed 0, fold 0');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(SUB/'evidence/backbones_validation.png',dpi=160);plt.close(fig)
    lines=['# Kết quả sàng lọc backbone','',
           '| exp_id | Macro-F1 val | Top-1 val | Train/epoch (s) | p50 sơ bộ (ms) |',
           '|---|---:|---:|---:|---:|']
    lines += [f"| {r['exp_id']} | {r['macro_f1_val']:.4f} | {r['top1_val']:.4f} | {r['training_seconds_per_epoch']:.2f} | {r['batch1_latency_ms']:.2f} |" for r in rows]
    lines += ['',f"Chọn **{chosen['exp_id']}**, macro-F1 val **{chosen['val']['macro_f1']:.4f}** để tiếp tục ablation.",
              '',choice['reason'],'',
              'Cùng 10 epoch, batch 32, 128×128, AdamW, CE, crop + flip, warmup + cosine và head normal(0,0,01).',
              'Mọi điểm chọn mô hình lấy từ toàn bộ val. Các tag tiền huấn luyện khác nhau được ghi rõ; thứ hạng này không cô lập riêng ảnh hưởng kiến trúc khỏi công thức tiền huấn luyện.',
              'Test chưa được suy luận. Các thời gian epoch lấy từ history.csv; không gồm tải trọng số hay ghi biểu đồ/checkpoint.']
    (SUB/'evidence/backbone_selection.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(choice,ensure_ascii=False),flush=True)
    return choice


def main():
    ap=argparse.ArgumentParser(description='Quy trình Lab Day 2 theo từng giai đoạn')
    ap.add_argument('stage',choices=['backbones','audit-backbones'])
    args=ap.parse_args()
    if args.stage=='backbones': backbones()
    elif args.stage=='audit-backbones': print(json.dumps(audit_backbones(),ensure_ascii=False))


if __name__=='__main__':main()
