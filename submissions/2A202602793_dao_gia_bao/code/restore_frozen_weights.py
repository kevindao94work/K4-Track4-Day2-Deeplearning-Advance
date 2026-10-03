"""Khôi phục cache task-local đúng snapshot/SHA đã chốt, không sửa recipe."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download,constants
from paths import SUB
from train import write_json


def restore():
    frozen=json.loads((SUB/'configs/frozen_final.json').read_text());sources={}
    for spec in frozen['configs'].values():
        source=spec['pretrained_source'];sources[(source['repo_id'],source['revision'])]=source
    evidence=[]
    for source in sources.values():
        cached=Path(hf_hub_download(source['repo_id'],source['filename'],revision=source['revision']))
        actual=hashlib.sha256(cached.read_bytes()).hexdigest()
        if actual!=source['sha256']:raise ValueError('Trọng số tải lại không khớp SHA chốt')
        # Factory timm đọc ref main: trong cache riêng của task, ref trỏ snapshot chốt.
        repo_cache=Path(constants.HF_HUB_CACHE)/('models--'+source['repo_id'].replace('/','--'))
        (repo_cache/'refs').mkdir(parents=True,exist_ok=True)
        (repo_cache/'refs/main').write_text(source['revision'])
        evidence.append({**source,'actual_sha256':actual,'cache':str(constants.HF_HUB_CACHE),'verified':True})
    write_json(SUB/'evidence/frozen_weights_restored.json',{'sources':evidence,
        'note':'Cache toàn cục mất trước T00 seed0; tải snapshot cố định sang cache data/hf_cache. Recipe, head init và trọng số nguồn không đổi; F01 giữ kết quả đã hoàn tất. Test chưa chạy.'})
    print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':restore()
