"""Kiểm tra giai đoạn 1 trên toàn bộ dữ liệu thật, lưu bằng chứng JSON."""
import argparse
import json
from pathlib import Path

import torch

from dataset import load_split, check_split, build_transforms, make_loader, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-dir', default='data/labels')
    ap.add_argument('--images-dir', default='data/images')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'evidence/stage1.json'))
    args = ap.parse_args()
    set_seed(0)
    train, val, test = load_split(args.labels_dir)
    evidence = check_split(train, val, test, args.images_dir)
    loader = make_loader(train, args.images_dir, build_transforms(True), 8, True, num_workers=0)
    x, y, filenames = next(iter(loader))
    assert x.ndim == 4 and y.ndim == 1 and x.shape[0] == y.shape[0]
    assert torch.isfinite(x).all() and ((y >= 0) & (y < 9)).all()
    t = build_transforms(False)
    a = next(iter(make_loader(val, args.images_dir, t, 8, False, num_workers=0)))
    b = next(iter(make_loader(val, args.images_dir, t, 8, False, num_workers=0)))
    assert torch.equal(a[0], b[0]) and a[2] == b[2]
    set_seed(7)
    c = next(iter(make_loader(train, args.images_dir, build_transforms(True), 8, True, num_workers=2, seed=7)))
    set_seed(7)
    d = next(iter(make_loader(train, args.images_dir, build_transforms(True), 8, True, num_workers=2, seed=7)))
    assert torch.equal(c[0], d[0]) and c[2] == d[2]
    try:
        load_split(args.labels_dir, fold=1)
    except ValueError:
        pass
    else:
        raise AssertionError('Chấp nhận sai fold')
    evidence['smoke'] = {'batch_shape': list(x.shape), 'all_finite': True,
                         'val_deterministic': True, 'seed_repeatable': True,
                         'filenames': list(filenames)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
