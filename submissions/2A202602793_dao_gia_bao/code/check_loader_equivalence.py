"""Kiểm tra tối ưu truy cập cột không đổi ảnh/nhãn/augmentation so với iloc cũ."""
import json
import hashlib
import time
import torch
from PIL import Image
from dataset import load_split, DeepWeedsDataset, build_transforms, set_seed
from train import Config, frozen_recipe, write_json
from paths import DATA, SUB


def legacy_item(ds,i):
    row=ds.df.iloc[i]
    with Image.open(ds.images_dir/row.Filename) as im: image=im.convert('RGB')
    return ds.transform(image),int(row.Label),str(row.Filename)


def main():
    train,val,_=load_split(DATA/'labels')
    results=[]
    for split,frame in [('train',train),('val',val)]:
        ds=DeepWeedsDataset(frame,DATA/'images',build_transforms(split=='train',128))
        indices=[int(frame.index[frame.Label==c][0]) for c in range(9)]
        max_error=0.
        for i in indices:
            set_seed(17+i);old=legacy_item(ds,i)
            set_seed(17+i);new=ds[i]
            if old[1:]!=new[1:]: raise AssertionError('Nhãn/tên thay đổi')
            max_error=max(max_error,float((old[0]-new[0]).abs().max()))
            if not torch.equal(old[0],new[0]): raise AssertionError('Tensor/augmentation thay đổi')
        results.append({'split':split,'images_checked':len(indices),'max_tensor_error':max_error})
    ds=DeepWeedsDataset(val,DATA/'images',build_transforms(False,128))
    timings={}
    for label,getter in [('iloc_with_attrs',lambda i:legacy_item(ds,i)),('cached_columns',ds.__getitem__)]:
        set_seed(91);t=time.perf_counter()
        for i in range(100):getter(i)
        timings[label]=time.perf_counter()-t
    evidence={'passed':True,'checks':results,'cpu_100_images_seconds':timings,
              'note':'Cùng PIL RGB, transform, nhãn và thứ tự; chỉ bỏ iloc gây deepcopy metadata. Đo CPU I/O, không phải latency GPU.'}
    reference=SUB/'predictions/B02_convnext_atto_seed0_val.csv'
    optimized=SUB/'predictions/T00_screen_seed0_val.csv'
    if reference.is_file() and optimized.is_file():
        configs=[Config(**json.loads((SUB/'logs'/eid/'seed0/summary.json').read_text())['config'])
                 for eid in ['B02_convnext_atto','T00_screen']]
        if frozen_recipe(configs[0])==frozen_recipe(configs[1]):
            a,b=reference.read_bytes(),optimized.read_bytes()
            if a!=b: raise AssertionError('Dự đoán sau 10 epoch không khớp tham chiếu')
            evidence['full_training_equivalence']={'reference':'B02_convnext_atto_seed0',
                'optimized':'T00_screen_seed0','epochs':10,'identical_val_prediction_csv':True,
                'csv_sha256':hashlib.sha256(a).hexdigest()}
    write_json(SUB/'evidence/loader_equivalence.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':main()
