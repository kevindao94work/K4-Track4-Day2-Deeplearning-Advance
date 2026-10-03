"""Chốt lựa chọn validation và xác minh bất biến trước khi mở test."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from paths import SUB,DATA,CODE,REPO
from train import Config,frozen_recipe,validate_freeze,write_json
from inference import Method,predict_method,fit_temperature,apply_temperature
import inference
from inference_lab import load_checkpoint,eval_loader
from eval import compute_metrics,read_pred,check_against_csv


def inference_fingerprint():
    names=['Method','method_logits','predict_method','apply_temperature','fit_temperature',
           'view_hflip','views_multicrop','views_multiscale','_logits']
    source='\n'.join(inspect.getsource(getattr(inference,n)) for n in names)
    return hashlib.sha256(source.encode()).hexdigest()


def frozen_path():return SUB/'configs/frozen_final.json'


def config_for(exp_id,seed):
    record=json.loads(frozen_path().read_text())
    cfg=Config(**record['configs'][exp_id]['recipe'],exp_id=exp_id,seed=seed,
        images_dir=str(DATA/'images'),labels_dir=str(DATA/'labels'),out_dir=str(SUB/'logs'),
        pred_dir=str(SUB/'predictions'),curves_dir=str(SUB/'curves'),freeze_file=str(frozen_path()))
    validate_freeze(cfg)
    return cfg


def assert_integrity(record):
    if record['inference_fingerprint']!=inference_fingerprint():raise ValueError('Phép suy luận đã đổi sau chốt')
    for name,sha in record['pipeline_sha256'].items():
        if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=sha:raise ValueError(f'{name} đã đổi sau chốt')
    for name,sha in record['csv_sha256'].items():
        if hashlib.sha256((DATA/'labels'/name).read_bytes()).hexdigest()!=sha:raise ValueError('CSV split đã đổi')
    if hashlib.sha256((REPO/'eval.py').read_bytes()).hexdigest()!=record['evaluator_sha256']:
        raise ValueError('eval.py đã đổi')


def freeze():
    if frozen_path().is_file():
        record=json.loads(frozen_path().read_text());assert_integrity(record)
        gate_path=SUB/'evidence/stage9.json'
        if gate_path.is_file():
            gate=json.loads(gate_path.read_text())
            if gate.get('passed') and gate.get('frozen_sha256')==hashlib.sha256(frozen_path().read_bytes()).hexdigest(): return record
        return dry_run(record)
    gate=json.loads((SUB/'evidence/stage8.json').read_text())
    if not gate['passed'] or gate['test_evaluated']:raise ValueError('Cổng val-only chưa đạt')
    training=json.loads((SUB/'configs/training_choice.json').read_text())
    infer=json.loads((SUB/'configs/inference_choice.json').read_text())
    baseline=json.loads((SUB/'logs/T00_screen/seed0/summary.json').read_text())
    selected=json.loads((Path(training['checkpoint']).parent/'summary.json').read_text())
    configs={}
    for eid,summary,method,policy in [('F01',selected,infer['method'],infer['temperature_policy']),
                                   ('T00',baseline,Method().__dict__,'none')]:
        configs[eid]={'recipe':frozen_recipe(Config(**summary['config'])),
            'inference':method,'temperature_policy':policy,
            'pretrained_source':summary['model']['pretrained_source'],
            'screening_source_exp_id':summary['exp_id'],
            'test_paths':{str(s):str(SUB/'predictions'/f'{eid}_seed{s}_test.csv') for s in [0,1,2]},
            'val_paths':{str(s):str(SUB/'predictions'/f'{eid}_seed{s}_val.csv') for s in [0,1,2]}}
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_split':'val',
        'seeds':[0,1,2],'configs':configs,'final_uncal_test_pattern':'F01_uncal_seed*_test.csv',
        'reason':'Backbone B02; công thức T05; suy luận I08. Chọn macro-F1 val; tie dùng NLL rồi p95.',
        'inference_fingerprint':inference_fingerprint(),
        'pipeline_sha256':{n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in ['dataset.py','model.py','losses.py','train.py']},
        'csv_sha256':json.loads((SUB/'evidence/stage5.json').read_text())['split']['csv_sha256'],
        'evaluator_sha256':hashlib.sha256((REPO/'eval.py').read_bytes()).hexdigest(),
        'test_policy':'Một vòng đầy đủ qua test mỗi exp_id/seed; lưu uncal/cal từ cùng logits; ledger chống chạy lại.',
        'temperature_procedure':'Mỗi seed tối ưu một log(T) trên NLL của toàn bộ val ở policy đã chốt; giới hạn [-5,5], không đổi argmax.',
        'screening_results':{'training':training['macro_f1_val'],'inference':infer['result']}}
    write_json(frozen_path(),record);assert_integrity(record)
    for eid in ['F01','T00']:
        for seed in record['seeds']:config_for(eid,seed)
    return dry_run(record)


def dry_run(record):
    configs=record['configs']
    selected=json.loads((SUB/'logs'/configs['F01']['screening_source_exp_id']/'seed0/summary.json').read_text())
    # Dry run đầy đủ val trên checkpoint sàng lọc, cấu hình suy luận đã chốt.
    model,device,source_cfg=load_checkpoint(selected);method=Method(**configs['F01']['inference'])
    names,y,z=predict_method(model,eval_loader(source_cfg,method),device,method)
    t=fit_temperature(z,y) if configs['F01']['temperature_policy']!='none' else 1.
    p=apply_temperature(z,t);metrics=compute_metrics(y,p.argmax(1),p)
    if len(names)!=3501 or abs(metrics['macro_f1']-record['screening_results']['inference']['macro_f1_val'])>1e-12:
        raise ValueError('Dry run không khớp cấu hình chọn')
    base=read_pred(SUB/'predictions/T00_screen_seed0_val.csv');check_against_csv(base,DATA/'labels/val_subset0.csv','val')
    write_json(SUB/'evidence/stage9.json',{'passed':True,'dry_run_split':'val','n':len(names),
        'temperature':t,'metrics':metrics,'baseline_val_n':len(base.filenames),
        'frozen_sha256':hashlib.sha256(frozen_path().read_bytes()).hexdigest(),'test_evaluated':False})
    (SUB/'evidence/frozen_configuration.md').write_text(
        '# Cấu hình chốt trước test\n\nF01: ConvNeXt-Atto d2_in1k, finetune, 10 epoch, batch 32, train 128×128, crop+flip, label smoothing 0,1, AdamW LR backbone/head 1e-4/1e-3, WD 0,05, warmup 1 epoch + cosine, FP32, không EMA.\n\nSuy luận: resize/center-crop 160, bicubic, crop_pct 0,875, một view; mỗi seed khớp một nhiệt độ trên NLL validation rồi áp dụng nguyên trạng sang test.\n\nMốc T00: cùng backbone và lịch train, CE, infer một view 128, không hiệu chỉnh. Cả hai dùng các seed 0,1,2, đủ 10 epoch và chọn checkpoint bằng macro-F1 val ở độ phân giải train. Sau đó policy inference cố định; không chọn lại checkpoint bằng val160/test.\n\nCác đường dẫn và hash ở configs/frozen_final.json. Dry run chỉ trên val; test chưa được suy luận. Cấu hình này được commit/push trước khi train chung kết và trước khi mở kết quả test.\n')
    return record


if __name__=='__main__':freeze()
