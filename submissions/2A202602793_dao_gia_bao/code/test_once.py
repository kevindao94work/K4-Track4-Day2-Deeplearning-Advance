"""Một vòng test mỗi ID/seed; lần gọi sau chỉ xác minh CSV đã có."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from paths import SUB,DATA
from train import validate_freeze,run_dir,write_json
from freeze_final import frozen_path,assert_integrity,config_for
from inference import Method,predict_method,apply_temperature
from inference_lab import load_checkpoint,eval_loader
from dataset import load_split,check_split
from eval import save_predictions,read_pred,check_against_csv,compute_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def begin_pass(path,metadata):
    """Tạo ledger độc quyền trước forward; lỗi giữa chừng không cho chạy lại."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    record={**metadata,'status':'started','pass_count':1,'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with path.open('x') as handle:json.dump(record,handle,ensure_ascii=False,indent=2)
    except FileExistsError as exc:raise ValueError('Ledger đã tồn tại; không được forward test lần nữa') from exc
    return record


def validate_cached(record,cfg,freeze_sha,checkpoint_sha):
    if record['status']!='complete':raise ValueError('Test đã bắt đầu nhưng chưa hoàn tất; cấm tự chạy lại')
    if record['pass_count']!=1 or record['frozen_sha256']!=freeze_sha or record['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('Ledger khác cấu hình/checkpoint chốt')
    for filename,digest in record['prediction_sha256'].items():
        path=Path(cfg.pred_dir)/filename
        if sha(path)!=digest:raise ValueError('CSV test đã bị sửa')
        pred=read_pred(path);check_against_csv(pred,Path(cfg.labels_dir)/'test_subset0.csv','test')
        if len(pred.filenames)!=record['n']:raise ValueError('CSV test thiếu ảnh')
    return record


def export_frozen_test(cfg):
    frozen=validate_freeze(cfg);assert_integrity(frozen)
    frozen_sha=sha(cfg.freeze_file)
    gate=json.loads((SUB/'evidence/stage10.json').read_text())
    if not gate['passed'] or gate['frozen_sha256']!=frozen_sha or len(gate['seeds'])<3:
        raise ValueError('Chưa đủ huấn luyện/validation chung kết')
    folder=run_dir(cfg);summary=json.loads((folder/'summary.json').read_text())
    val=json.loads((folder/'final_val.json').read_text());ledger=folder/'test_ledger.json'
    if ledger.is_file():
        return validate_cached(json.loads(ledger.read_text()),cfg,frozen_sha,summary['checkpoint_sha256'])
    if Path(cfg.pred_dir).resolve()!=(SUB/'predictions').resolve():raise ValueError('Đường dẫn xuất khác kế hoạch')
    if val['frozen_sha256']!=frozen_sha or val['checkpoint_sha256']!=summary['checkpoint_sha256']:
        raise ValueError('Nhiệt độ không thuộc checkpoint/cấu hình chốt')
    expected={(e,s) for e in ['F01','T00'] for s in frozen['seeds']}
    if {(r['exp_id'],r['seed']) for r in gate['runs']}!=expected:raise ValueError('Thiếu seed/mốc')
    for eid,seed in expected:
        c=config_for(eid,seed);f=run_dir(c)
        s=json.loads((f/'summary.json').read_text());v=json.loads((f/'final_val.json').read_text())
        if s['epochs_completed']!=c.epochs or v['val']['n']!=3501 or v['frozen_sha256']!=frozen_sha:
            raise ValueError('Seed chưa hoàn tất train/val')
        if sha(s['best_checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Checkpoint đã đổi')
        p=read_pred(SUB/'predictions'/f'{eid}_seed{seed}_val.csv')
        check_against_csv(p,DATA/'labels/val_subset0.csv','val')
        if sha(p.path)!=v['prediction_sha256']:raise ValueError('Val CSV đã đổi sau khớp T')
    train,val_df,test=load_split(cfg.labels_dir);check_split(train,val_df,test,cfg.images_dir)
    method=Method(**frozen['configs'][cfg.exp_id]['inference'])
    model,device,_=load_checkpoint(summary);loader=eval_loader(cfg,method,'test')
    metadata={'exp_id':cfg.exp_id,'seed':cfg.seed,'frozen_sha256':frozen_sha,
              'checkpoint_sha256':summary['checkpoint_sha256'],'temperature':val['temperature'],
              'temperature_fit_split':val['temperature_fit_split'],'n':3507,'method':frozen['configs'][cfg.exp_id]['inference']}
    state=begin_pass(ledger,metadata)
    names,y,z=predict_method(model,loader,device,method)  # đúng một vòng đầy đủ
    if len(names)!=3507:raise ValueError('Test phải đủ 3.507 ảnh')
    np.savez(folder/'test_outputs.npz',filenames=np.asarray(names),y_true=y,logits=z)
    raw=apply_temperature(z,1.);cal=apply_temperature(z,val['temperature'])
    if not np.array_equal(raw.argmax(1),cal.argmax(1)):raise ValueError('TS đổi argmax test')
    filename=f'{cfg.exp_id}_seed{cfg.seed}_test.csv'
    uncal_name=f'{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv'
    hashes={}
    for name,p in [(filename,cal),(uncal_name,raw)]:
        path=Path(cfg.pred_dir)/name;save_predictions(path,names,y,p)
        pred=read_pred(path);check_against_csv(pred,Path(cfg.labels_dir)/'test_subset0.csv','test')
        hashes[name]=sha(path)
    state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),prediction_sha256=hashes,
                 argmax_unchanged=True,forward_dataset_passes=1)
    write_json(ledger,state)
    summary['test_evaluated']=True;write_json(folder/'summary.json',summary)
    return state


def export_all():
    records=[]
    for eid in ['F01','T00']:
        for seed in [0,1,2]:
            record=export_frozen_test(config_for(eid,seed));records.append(record)
            print(json.dumps({'exp_id':eid,'seed':seed,'n':record['n'],'pass_count':record['pass_count'],'status':record['status']},ensure_ascii=False),flush=True)
    write_json(SUB/'evidence/test_passes.json',{'records':records,'all_once':all(r['pass_count']==1 for r in records)})
    return records


if __name__=='__main__':export_all()
