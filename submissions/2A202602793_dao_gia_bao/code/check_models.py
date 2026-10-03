"""Cổng model: forward, đóng băng, nhóm tham số, thống kê đo tại chỗ."""
import json
import argparse
from pathlib import Path

import torch
from torch import nn

from model import SCREENING_MODELS, build_model, classifier, param_groups, model_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = []
    for name in SCREENING_MODELS:
        for init in ['finetune', 'frozen']:
            m = build_model(name, pretrained=args.pretrained, init=init, img_size=128)
            m.train()
            x = torch.randn(2, 3, 128, 128)
            y = m(x)
            assert y.shape == (2, 9) and torch.isfinite(y).all()
            head_ids = {id(p) for p in classifier(m).parameters()}
            if init == 'frozen':
                assert all(p.requires_grad == (id(p) in head_ids) for p in m.parameters())
                assert all(not b.training for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm))
            groups = param_groups(m, 1e-4, 1e-3, .05)
            ids = [id(p) for g in groups for p in g['params']]
            assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in m.parameters())
            norm_ids = {id(p) for b in m.modules() if isinstance(b, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm))
                        for p in b.parameters(recurse=False)}
            for g in groups:
                for pname, p in zip(g['param_names'], g['params']):
                    assert g['lr'] == (1e-3 if id(p) in head_ids else 1e-4)
                    if pname.endswith('.bias') or id(p) in norm_ids or p.ndim <= 1:
                        assert g['weight_decay'] == 0
                    elif pname.endswith('.weight'):
                        assert g['weight_decay'] == .05
            info = model_metadata(m, 128)
            info.update(init=init, shape=list(y.shape), finite=True,
                        tested_pretrained=args.pretrained, grouping_valid=True)
            rows.append(info)
            print(name, init, info['parameters_m'], info['gmac'], flush=True)
    out = Path(__file__).resolve().parents[1] / ('evidence/stage2_pretrained.json' if args.pretrained else 'evidence/stage2.json')
    if args.out:
        out = Path(args.out)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
