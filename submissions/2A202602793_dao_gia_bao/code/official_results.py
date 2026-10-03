"""Đánh giá từ CSV đã có; không gọi model hoặc tạo lại dự đoán test."""
import json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import SUB,DATA,REPO
from train import write_json
from freeze_final import frozen_path,assert_integrity
from eval import load_group,load_names,SCALARS,mean_std


def official_results():
    frozen=json.loads(frozen_path().read_text());assert_integrity(frozen)
    passes=json.loads((SUB/'evidence/test_passes.json').read_text())
    if not passes['all_once'] or len(passes['records'])!=6 or any(r['status']!='complete' or r['n']!=3507 for r in passes['records']):
        raise ValueError('Test chưa đủ sáu lượt đúng một lần')
    output=SUB/'evidence/official_eval';output.mkdir(parents=True,exist_ok=True)
    groups={}
    for eid in ['F01','T00','F01_uncal']:
        pattern=str(SUB/'predictions'/f'{eid}_seed*_test.csv')
        args=[sys.executable,str(REPO/'eval.py'),'score','--pred',pattern,'--test-csv',str(DATA/'labels/test_subset0.csv'),
              '--labels',str(DATA/'labels/labels.csv'),'--tag',eid,'--out',str(output)]
        result=subprocess.run(args,text=True,capture_output=True,check=True)
        (output/f'{eid}_score.md').write_text('# Bộ đánh giá chính thức\n\n'+result.stdout)
        groups[eid]=load_group([pattern],str(DATA/'labels/test_subset0.csv'))
    latency=json.loads((SUB/'logs/F01/seed0/batch1_latency.json').read_text())
    args=[sys.executable,str(REPO/'eval.py'),'grade','--final',str(SUB/'predictions/F01_seed*_test.csv'),
          '--baseline',str(SUB/'predictions/T00_seed*_test.csv'),'--uncal',str(SUB/'predictions/F01_uncal_seed*_test.csv'),
          '--final-val',str(SUB/'predictions/F01_seed*_val.csv'),'--test-csv',str(DATA/'labels/test_subset0.csv'),
          '--val-csv',str(DATA/'labels/val_subset0.csv'),'--labels',str(DATA/'labels/labels.csv'),
          '--latency-p95-ms',str(latency['p95']),'--latency-method','proper','--out',str(output)]
    result=subprocess.run(args,text=True,capture_output=True,check=True)
    (output/'grade.md').write_text('# Tự chấm bằng evaluator nguyên bản\n\n'+result.stdout)
    print(result.stdout,flush=True)
    names=load_names(str(DATA/'labels/labels.csv'));val=pd.read_csv(SUB/'tables/FinalValidation.csv')
    rows=[];perclass=[];stats={}
    for eid in ['F01','T00']:
        group=groups[eid];spec=frozen['configs'][eid]
        config_text=(f"{spec['recipe']['backbone']}; finetune; train128; 10ep/b32; {spec['recipe']['loss']}"
            f" LS={spec['recipe']['label_smoothing']}; crop+flip; AdamW LR1e-4/1e-3 WD0.05 warmup1+cos;"
            f" FP32; noEMA; infer{spec['inference']['resolution']} K1; {spec['temperature_policy']}")
        stats[eid]={k:{'mean':group.summary[k][0],'std':group.summary[k][1]} for k in SCALARS}
        for pred,metrics in zip(group.preds,group.metrics):
            vr=val[(val.exp_id==eid)&(val.seed==pred.seed)].iloc[0]
            rows.append({'exp_id':eid,'complete_configuration':config_text,'seed':pred.seed,
                'macro_f1_val':vr.macro_f1_val,'macro_f1_test':metrics['macro_f1'],'top1_test':metrics['top1'],
                'ece_test':metrics['ece'],'nll_test':metrics['nll'],'temperature_from_val':vr.temperature,
                'mean_std_summary':f"F1 {group.summary['macro_f1'][0]:.4f} ± {group.summary['macro_f1'][1]:.4f}; Top1 {group.summary['top1'][0]:.4f} ± {group.summary['top1'][1]:.4f}",
                'macro_f1_test_mean':group.summary['macro_f1'][0],'macro_f1_test_std':group.summary['macro_f1'][1]})
        for i,name in enumerate(names):
            perclass.append({'exp_id':eid,'class_label':i,'class':name,'test_count':int(group.metrics[0]['support'][i]),
                **{k:float(group.summary[k][0][i]) for k in ['precision','recall','f1']},
                **{k+'_std':float(group.summary[k][1][i]) for k in ['precision','recall','f1']}})
    pd.DataFrame(rows).to_csv(SUB/'tables/Final.csv',index=False)
    pd.DataFrame(perclass).to_csv(SUB/'tables/PerClass.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(17,7))
    for ax,eid in zip(axes,['T00','F01']):
        cm=sum(m['confusion'] for m in groups[eid].metrics)
        normalized=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
        for i in range(9):
            for j in range(9):
                ax.text(j,i,f'{cm[i,j]}\n{normalized[i,j]*100:.1f}%',ha='center',va='center',fontsize=6,color='white' if normalized[i,j]>.5 else 'black')
        ax.set_xticks(range(9),names,rotation=55,ha='right',fontsize=8);ax.set_yticks(range(9),names,fontsize=8)
        ax.set_xlabel('Nhãn dự đoán');ax.set_ylabel('Nhãn thật');ax.set_title(eid+' — cộng ba seed, 10.521 dự đoán')
    fig.suptitle('Test 3.507 ảnh/seed — màu chuẩn hóa theo nhãn thật; số đếm cộng ba seed')
    fig.tight_layout();fig.savefig(SUB/'evidence/confusion_test.png',dpi=160);plt.close(fig)
    errors=[];pair_counts=[]
    for pred in groups['F01'].preds:
        wrong=pred.y_true!=pred.y_pred
        pair=wrong&np.isin(pred.y_true,[0,7])&np.isin(pred.y_pred,[0,7])
        pair_counts.append({'seed':pred.seed,'chinee_to_snake':int(((pred.y_true==0)&(pred.y_pred==7)).sum()),
                            'snake_to_chinee':int(((pred.y_true==7)&(pred.y_pred==0)).sum()),'errors_total':int(wrong.sum())})
        for i in np.flatnonzero(wrong):
            errors.append({'seed':pred.seed,'filename':str(pred.filenames[i]),'true_label':int(pred.y_true[i]),
                           'pred_label':int(pred.y_pred[i]),'confidence':float(pred.probs[i].max()),
                           'pair_0_7':bool(pair[i]),'focus':bool(pred.y_true[i] in (0,7))})
    errors.sort(key=lambda e:(not e['pair_0_7'],not e['focus'],e['seed'],e['filename']))
    chosen=[];seen=set()
    for error in errors:
        if error['filename'] not in seen:chosen.append(error);seen.add(error['filename'])
        if len(chosen)==8:break
    fig,axes=plt.subplots(2,4,figsize=(16,10),layout='constrained')
    for ax,error in zip(axes.flat,chosen):
        with Image.open(DATA/'images'/error['filename']) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(f"seed {error['seed']} · {error['filename']}\nThật: {names[error['true_label']]}\nDự đoán: {names[error['pred_label']]} ({error['confidence']:.2f})",fontsize=8)
        ax.axis('off')
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    fig.suptitle('Lỗi test đại diện — phân tích sau chốt, không dùng để điều chỉnh mô hình')
    fig.savefig(SUB/'evidence/error_examples_test.png',dpi=160);plt.close(fig)
    write_json(SUB/'evidence/error_examples_test.json',{'examples':chosen,'pair_counts':pair_counts,
        'selection':'Ưu tiên nhầm Chinee↔Snake trên ba seed, rồi lớp thật 0/7; tên ảnh duy nhất; tất cả điểm số vẫn dùng đầy đủ test.'})
    paired=[a['macro_f1']-b['macro_f1'] for a,b in zip(groups['F01'].metrics,groups['T00'].metrics)]
    dm,ds=mean_std(paired)
    gate={'passed':True,'ddof':1,'test_n_per_seed':3507,'groups':stats,
          'paired_delta_macro_f1':{'mean':dm,'std':ds,'per_seed':paired},'test_passes_per_seed':1,
          'uncal_ece':dict(zip(['mean','std'],groups['F01_uncal'].summary['ece'])),
          'final_latency_p95_batch1_ms':latency['p95'],'temperature_fit_split':'val',
          'configuration_unchanged_after_test':True,'official_output':str(output)}
    write_json(SUB/'evidence/stage11.json',gate)
    return gate


if __name__=='__main__':official_results()
