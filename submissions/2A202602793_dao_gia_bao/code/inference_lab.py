"""So sánh suy luận trên toàn bộ val và đo độ trễ thật từng policy."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from paths import SUB,DATA
from train import Config,run_dir,resolve_device,write_json,validate_freeze
from model import build_model
from dataset import load_split,build_transforms,make_loader
from inference import Method,predict_method,fit_temperature,apply_temperature
from benchmark import method_latency
from eval import compute_metrics,save_predictions,read_pred,check_against_csv


def load_checkpoint(summary):
    cfg=Config(**summary['config']);device=resolve_device(cfg.device)
    checkpoint=Path(summary['best_checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=summary['checkpoint_sha256']:
        raise ValueError('Checkpoint đã đổi so với summary')
    model=build_model(cfg.backbone,False,init=cfg.init,img_size=cfg.img_size).to(device).eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    return model,device,cfg


def eval_loader(cfg,method,split='val'):
    if split not in ('val','test'):raise ValueError('Chỉ loader đánh giá')
    if split=='test':validate_freeze(cfg)
    frames=load_split(cfg.labels_dir)
    df=frames[1 if split=='val' else 2]
    record=json.loads((run_dir(cfg)/'config.json').read_text())
    pre=record['preprocessing']
    transform=build_transforms(False,method.input_size,mean=pre['mean'],std=pre['std'],
                               interpolation=pre['interpolation'],crop_pct=pre['crop_pct'])
    return make_loader(df,cfg.images_dir,transform,cfg.batch_size,False,num_workers=cfg.num_workers,seed=cfg.seed)


def export_val(names,y,logits,temperature,eid,folder):
    p=apply_temperature(logits,temperature)
    path=SUB/'predictions'/f'{eid}_seed0_val.csv'
    save_predictions(path,names,y,p);pred=read_pred(path)
    check_against_csv(pred,DATA/'labels/val_subset0.csv','val')
    m=compute_metrics(pred.y_true,pred.y_pred,pred.probs)
    write_json(folder/f'{eid}_metrics.json',m)
    return m


def run_inference():
    if not json.loads((SUB/'evidence/stage7.json').read_text())['passed']:raise ValueError('Chưa đạt ablation')
    selected=json.loads((SUB/'configs/training_choice.json').read_text())
    summary=json.loads((Path(selected['checkpoint']).parent/'summary.json').read_text())
    model,device,cfg=load_checkpoint(summary)
    folder=SUB/'logs/inference';folder.mkdir(parents=True,exist_ok=True)
    methods=[Method(),Method('I01','Lật ngang, trung bình xác suất','hflip',aggregation='prob'),
             Method('I02','Lật ngang, trung bình logit','hflip'),
             Method('I03','Năm crop 128 từ ảnh 160','fivecrop',input_size=160,aggregation='prob'),
             Method('I04','Một view 160','identity',160,160),
             Method('I05','Một view 224','identity',224,224),
             Method('I06','Ba scale 96/128/160','multiscale',aggregation='prob')]
    rows=[];latencies=[];outputs={};measured={}
    def measure(method,t,eid,label,metrics):
        pairs=[]
        for batch in (1,32):
            latency=method_latency(model,method,device,batch,t,iters=50)
            latency['configuration']=eid;latency['method']=label;latency['temperature']=t
            write_json(folder/f'{eid}_batch{batch}_latency.json',latency)
            latencies.append({k:v for k,v in latency.items() if k!='samples_ms'});pairs.append(latency)
        one,throughput=pairs
        row={'exp_id':eid,'method':label,'model_checkpoint':selected['source_exp_id']+'_seed0/best.pt',
             'checkpoint_sha256':summary['checkpoint_sha256'],'K':method.k,
             'resolution':method.resolution,'input_size':method.input_size,'temperature':t,
             'macro_f1_val':metrics['macro_f1'],'top1_val':metrics['top1'],'ece_val':metrics['ece'],
             'nll_val':metrics['nll'],'latency_p50_batch1_ms':one['p50'],
             'latency_p95_batch1_ms':one['p95'],'latency_p99_batch1_ms':one['p99'],
             'throughput_images_s':throughput['images_per_s'],
             'relative_cost_vs_I00':one['p50']/rows[0]['latency_p50_batch1_ms'] if rows else 1.}
        rows.append(row);measured[eid]=asdict(method)
        pd.DataFrame(rows).to_csv(SUB/'tables/Inference.csv',index=False)
        pd.DataFrame(latencies).to_csv(SUB/'tables/Latency.csv',index=False)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    for method in methods:
        names,y,logits=predict_method(model,eval_loader(cfg,method),device,method)
        outputs[method.exp_id]=(names,y,logits,method)
        np.savez(folder/f'{method.exp_id}_val_outputs.npz',filenames=names,y_true=y,logits=logits)
        metrics=export_val(names,y,logits,1.,method.exp_id,folder)
        measure(method,1.,method.exp_id,method.name,metrics)
    # Hai phép TS dùng logits val đã có, không thêm lần đọc hay forward test.
    raw_best=max(rows,key=lambda r:r['macro_f1_val'])['exp_id']
    calibration=[]
    for eid,source in [('I07','I00'),('I08',raw_best)]:
        names,y,logits,method=outputs[source]
        t=fit_temperature(logits,y);before=apply_temperature(logits,1.);after=apply_temperature(logits,t)
        if not np.array_equal(before.argmax(1),after.argmax(1)):raise ValueError('TS đổi argmax')
        raw=compute_metrics(y,before.argmax(1),before);cal=export_val(names,y,logits,t,eid,folder)
        if cal['nll']>raw['nll']+1e-7:raise ValueError('TS tăng NLL val')
        label=f'{method.name} + TS khớp NLL val'
        measure(method,t,eid,label,cal)
        calibration.append({'exp_id':eid,'source':source,'temperature':t,'fit_split':'val',
            'n':len(y),'argmax_unchanged':True,'ece_before':raw['ece'],'ece_after':cal['ece'],
            'nll_before':raw['nll'],'nll_after':cal['nll']})
    ordered=sorted(rows,key=lambda r:(-r['macro_f1_val'],r['nll_val'],r['latency_p95_batch1_ms']))
    best=ordered[0]
    choice={'source_exp_id':best['exp_id'],'method':measured[best['exp_id']],
            'temperature_policy':'val_nll_per_seed' if best['exp_id'] in ('I07','I08') else 'none',
            'screening_temperature':best['temperature'],'selection_split':'val','result':best,
            'reason':'Macro-F1 val cao nhất; khi hòa ưu tiên NLL val rồi p95 thấp. TS fit riêng val mỗi seed trước test.'}
    write_json(SUB/'configs/inference_choice.json',choice)
    gate={'passed':True,'val_n':3501,'warmups':10,'timed_iterations':50,'batches':[1,32],
          'synchronization':device.type,'calibration':calibration,
          'batchnorm_layers':sum(isinstance(m,nn.BatchNorm2d) for m in model.modules()),
          'bn_fusion_note':(f'{cfg.backbone}: không có BN để gộp; utility đã kiểm tra trên Conv/BN CPU.' if not any(isinstance(m,nn.BatchNorm2d) for m in model.modules()) else f'{cfg.backbone}: có BN nhưng không thực hiện ablation fusion; utility đã kiểm tra trên Conv/BN CPU.'),
          'methods':[r['exp_id'] for r in rows],'test_evaluated':False}
    write_json(SUB/'evidence/stage8.json',gate)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for row in rows:
        ax.scatter(row['latency_p95_batch1_ms'],row['macro_f1_val'])
        ax.annotate(row['exp_id'],(row['latency_p95_batch1_ms'],row['macro_f1_val']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('p95 batch 1 (ms, không tiền xử lý CPU)');ax.set_ylabel('Macro-F1 validation')
    ax.set_title('Đánh đổi chất lượng và độ trễ — một checkpoint');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(SUB/'evidence/inference_tradeoff.png',dpi=160);plt.close(fig)
    lines=['# So sánh suy luận trên validation','',
           '| ID | Phương pháp | K | Macro-F1 | ECE | p95 (ms) | Ảnh/s batch32 |',
           '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['exp_id']} | {r['method']} | {r['K']} | {r['macro_f1_val']:.4f} | {r['ece_val']:.4f} | {r['latency_p95_batch1_ms']:.2f} | {r['throughput_images_s']:.1f} |" for r in rows]
    lines += ['',f"Chọn {best['exp_id']}. "+choice['reason'],'',
        'Mọi phương pháp dùng cùng checkpoint và 3.501 ảnh val đúng thứ tự. Five-crop lấy bốn góc và tâm 128×128 trên ảnh tiền xử lý 160×160. Multi-scale nội suy bilinear không antialias tensor đã chuẩn hóa tới 96/128/160.',
        'Độ trễ có 10 warmup, 50 mẫu, đồng bộ MPS/CUDA trước và sau mỗi mẫu; gồm view/forward/gộp/softmax, chưa gồm đọc ảnh/resize/normalize CPU. Batch 1 và throughput batch 32 được đo độc lập.',
        'Temperature scaling giữ nguyên argmax, tối ưu NLL trên val, không bảo đảm ECE luôn giảm trên dữ liệu mới. Các nhiệt độ và ECE/NLL trước-sau ở stage8.json.',
        gate['bn_fusion_note'],'Test chưa được suy luận. Không chạy ensemble, EMA, FP16 hay AMP như thí nghiệm chất lượng; các utility/engine tương ứng không được coi là kết quả đo.']
    (SUB/'evidence/inference_selection.md').write_text('\n'.join(lines)+'\n')
    return choice


if __name__=='__main__':run_inference()
