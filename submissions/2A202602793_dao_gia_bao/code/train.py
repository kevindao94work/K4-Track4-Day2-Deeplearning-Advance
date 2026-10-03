"""Một engine cho mọi thí nghiệm; chọn checkpoint trên val, test có cổng chốt cấu hình."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import re
import sys
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn

from dataset import load_split, check_split, build_transforms, make_loader, set_seed
from losses import build_criterion, weights_from_train, mix_batch, mixed_loss
from model import build_model, model_metadata, param_groups

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'eval.py').is_file())
sys.path.insert(0, str(REPO))
from eval import compute_metrics, save_predictions, read_pred, check_against_csv


@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    init: str = 'finetune'
    pretrained: bool = True
    drop_rate: float = 0.
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.
    loss: str = 'ce'
    label_smoothing: float = 0.
    focal_gamma: float = 2.
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = .05
    warmup_epochs: float = 1.
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    save_test_predictions: bool = False
    curves_dir: str = 'curves'
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    momentum: float = .9
    grad_clip: float | None = None
    device: str = 'auto'
    crop_pct: float = .875
    interpolation: str = 'bicubic'
    resume: bool = True
    smoke_train: int = 0
    smoke_val: int = 0
    freeze_file: str | None = None


RUNTIME_FIELDS = {'exp_id', 'seed', 'out_dir', 'pred_dir', 'curves_dir', 'images_dir',
                  'labels_dir', 'save_test_predictions', 'resume', 'freeze_file'}


def frozen_recipe(cfg):
    return {k: v for k, v in asdict(cfg).items() if k not in RUNTIME_FIELDS}


def validate_freeze(cfg):
    if not cfg.freeze_file or not Path(cfg.freeze_file).is_file():
        raise ValueError('Chưa có cấu hình chốt trên val; test vẫn được niêm phong')
    frozen = json.loads(Path(cfg.freeze_file).read_text())
    if cfg.seed not in frozen['seeds'] or cfg.exp_id not in frozen['configs']:
        raise ValueError('Seed/exp_id không thuộc cấu hình đã chốt')
    if frozen_recipe(cfg) != frozen['configs'][cfg.exp_id]['recipe']:
        raise ValueError('Công thức chạy khác cấu hình đã chốt')
    return frozen


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'


def pred_path(cfg: Config, split: str) -> Path:
    if split not in ('val', 'test'):
        raise ValueError('Split dự đoán phải val hoặc test')
    return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'


def resolve_device(device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(device)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def hardware_name(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    if platform.system() == 'Darwin':
        name = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        return name + (' (MPS)' if device.type == 'mps' else ' (CPU)')
    return platform.processor() or 'CPU'


def versions(device):
    return {'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform(),
            'device': str(device), 'gpu': hardware_name(device),
            'threads': torch.get_num_threads(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'code_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ['dataset.py', 'model.py', 'losses.py', 'train.py']}}


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Không thể ghi JSON: {type(obj)}')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_safe, allow_nan=False))
    temp.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.stem + '.tmp' + path.suffix)
    torch.save(state, temp)
    temp.replace(path)


def build_optimizer(model, cfg: Config):
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(groups)
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(groups, momentum=cfg.momentum)
    raise ValueError('Optimizer phải adamw hoặc sgd')


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total = cfg.epochs * steps_per_epoch
    warmup = round(cfg.warmup_epochs * steps_per_epoch)
    if cfg.scheduler not in ('cosine', 'none') or total <= 0 or not 0 <= warmup < total:
        raise ValueError('Lịch LR cần số bước dương và warmup ngắn hơn tổng huấn luyện')

    def factor(step):
        if cfg.scheduler == 'none':
            return 1.
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1., (step - warmup) / (total - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay: float):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay phải nằm trong [0,1)')
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(source[name].detach(), alpha=1 - self.decay)
            else:
                value.copy_(source[name])

    def copy_to(self, model):
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        targets = y
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        enabled = cfg.amp and device.type == 'cuda'
        with torch.autocast(device_type=device.type, enabled=enabled):
            logits = model(x)
            loss = mixed_loss(criterion, logits.float(), targets) if cfg.mix else criterion(logits.float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Loss train không hữu hạn')
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = not scaler.is_enabled() or scaler.get_scale() >= old_scale
        if stepped:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        total_loss += loss.detach() * len(y)
        if not cfg.mix:
            correct += (logits.detach().argmax(1) == y).sum()
        n += len(y)
    if n == 0:
        raise ValueError('Train loader rỗng')
    return {'train_loss': total_loss.item() / n, 'train_top1': None if cfg.mix else correct.item() / n,
            'lr': max(g['lr'] for g in optimizer.param_groups), 'train_n': n}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    names, targets, outputs = [], [], []
    total = 0.
    for x, y, files in loader:
        x, y = x.to(device), y.to(device)
        z = model(x).float()
        loss = criterion(z, y)
        if not torch.isfinite(z).all() or not torch.isfinite(loss):
            raise FloatingPointError('Đầu ra val không hữu hạn')
        names.extend(files)
        targets.append(y.cpu().numpy())
        outputs.append(z.cpu().numpy())
        total += loss.item() * len(y)
    if not targets:
        raise ValueError('Loader đánh giá rỗng')
    labels, logits = np.concatenate(targets), np.concatenate(outputs)
    return names, labels, logits, total / len(labels)


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(h.epoch, h.train_loss, marker='o', label='Train (loss mục tiêu)')
    axes[0].plot(h.epoch, h.val_loss, marker='o', label='Val (CE)')
    axes[0].set_ylabel('Mất mát'); axes[0].legend()
    axes[1].plot(h.epoch, h.val_macro_f1, marker='o', label='Macro-F1 val')
    axes[1].plot(h.epoch, h.val_top1, marker='o', label='Top-1 val')
    axes[1].set_ylabel('Chỉ số'); axes[1].legend()
    axes[2].plot(h.epoch, h.lr, marker='o', label='LR cuối epoch')
    axes[2].set_ylabel('Learning rate'); axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rng_state(train_loader, val_loader):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
             'train_generator': train_loader.generator.get_state(), 'val_generator': val_loader.generator.get_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, train_loader, val_loader):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    train_loader.generator.set_state(state['train_generator']); val_loader.generator.set_state(state['val_generator'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def run(cfg: Config) -> dict:
    if cfg.fold != 0 or cfg.epochs <= 0 or not 0 <= cfg.seed < 2**32 or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', cfg.exp_id):
        raise ValueError('Fold/epochs/seed/exp_id không hợp lệ')
    if (cfg.smoke_train or cfg.smoke_val) and not cfg.exp_id.startswith('SMOKE'):
        raise ValueError('Chỉ dùng tập nhỏ với exp_id bắt đầu SMOKE')
    if cfg.freeze_file or cfg.save_test_predictions:
        validate_freeze(cfg)
    torch.set_num_threads(4)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / 'config.json'
    if config_file.exists():
        previous_record = json.loads(config_file.read_text())
        previous = previous_record['config']
        if {k:v for k,v in previous.items() if k not in ('resume','save_test_predictions')} != {k:v for k,v in asdict(cfg).items() if k not in ('resume','save_test_predictions')}:
            raise ValueError('exp_id/seed đã có cấu hình khác; hãy dùng mã thí nghiệm mới')
        if not cfg.resume:
            raise FileExistsError('Lần chạy đã tồn tại và resume=False')
        if (out / 'summary.json').is_file():
            summary = json.loads((out / 'summary.json').read_text())
            if cfg.save_test_predictions:
                from inference import export_frozen_test
                export_frozen_test(cfg)
                summary['test_evaluated'] = True
                write_json(out / 'summary.json', summary)
            return summary
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_info = check_split(train_df, val_df, test_df, cfg.images_dir)
    if config_file.exists() and previous_record['split']['csv_sha256'] != split_info['csv_sha256']:
        raise ValueError('CSV thay đổi so với lần chạy trước; không thể tiếp tục')
    model = build_model(cfg.backbone, cfg.pretrained, drop_rate=cfg.drop_rate, init=cfg.init, img_size=cfg.img_size)
    meta = model_metadata(model, cfg.img_size)
    preprocessing = {'mean': meta['pretrained_cfg']['mean'], 'std': meta['pretrained_cfg']['std'],
                     'interpolation': cfg.interpolation, 'crop_pct': cfg.crop_pct}
    effective = {'config': asdict(cfg), 'versions': versions(device), 'model': meta,
                 'preprocessing': preprocessing, 'split': split_info,
                 'amp_effective': cfg.amp and device.type == 'cuda',
                 'validation_loss': 'CE không trọng số',
                 'checkpoint_weights': 'EMA' if cfg.ema_decay is not None else 'raw'}
    write_json(config_file, effective)
    train_counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    weights = weights_from_train(train_df, cfg.class_weight_beta or 0.) if cfg.loss == 'ce_weighted' else None
    train_part = train_df.iloc[:cfg.smoke_train] if cfg.smoke_train else train_df
    val_part = val_df.iloc[:cfg.smoke_val] if cfg.smoke_val else val_df
    train_loader = make_loader(train_part, cfg.images_dir,
        build_transforms(True, cfg.img_size, cfg.aug, **preprocessing), cfg.batch_size,
        True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_part, cfg.images_dir,
        build_transforms(False, cfg.img_size, **preprocessing), cfg.batch_size,
        False, num_workers=cfg.num_workers, seed=cfg.seed)
    model.to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weights).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=effective['amp_effective'])
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_score, best_epoch, start = [], -1., None, 1
    latest = out / 'latest.pt'
    if latest.is_file():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        if ema is not None:
            ema.model.load_state_dict(state['ema'])
        history, best_score, best_epoch, start = state['history'], state['best_score'], state['best_epoch'], state['epoch'] + 1
        restore_rng(state['rng'], train_loader, val_loader)
    curve = Path(cfg.curves_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{model.lab_architecture}.png'
    for epoch in range(start, cfg.epochs + 1):
        synchronize(device); t0 = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        synchronize(device); t1 = time.perf_counter()
        _, labels, logits, val_loss = evaluate(ema.model if ema else model, val_loader, val_criterion, device)
        p = probabilities(logits)
        metrics = compute_metrics(labels, p.argmax(1), p)
        synchronize(device); seconds = time.perf_counter() - t0
        row = {'epoch': epoch, **train_stats, 'val_loss': val_loss, 'val_macro_f1': metrics['macro_f1'],
               'val_top1': metrics['top1'], 'val_balanced_acc': metrics['balanced_acc'], 'seconds': seconds,
               'train_seconds': t1 - t0, 'val_seconds': seconds - (t1 - t0)}
        if not all(math.isfinite(row[k]) for k in ['train_loss', 'val_loss', 'val_macro_f1', 'val_top1', 'seconds']):
            raise FloatingPointError('Chỉ số epoch không hữu hạn')
        history.append(row)
        if metrics['macro_f1'] > best_score:
            best_score, best_epoch = metrics['macro_f1'], epoch
            selected = ema.model if ema else model
            save_checkpoint(out / 'best.pt', {'model': selected.state_dict(), 'epoch': epoch,
                                              'macro_f1': best_score, 'config': asdict(cfg),
                                              'weights': 'EMA' if ema else 'raw'})
        save_checkpoint(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'ema': ema.model.state_dict() if ema else None,
            'epoch': epoch, 'history': history, 'best_score': best_score, 'best_epoch': best_epoch,
            'rng': rng_state(train_loader, val_loader)})
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        plot_curves(history, curve, f'{cfg.exp_id} · seed {cfg.seed} · {cfg.backbone}')
        print(json.dumps({'exp_id':cfg.exp_id, 'seed':cfg.seed, **row}, ensure_ascii=False), flush=True)
    # Khôi phục cả artifacts nếu gián đoạn ngay sau checkpoint epoch cuối.
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    plot_curves(history, curve, f'{cfg.exp_id} · seed {cfg.seed} · {cfg.backbone}')
    state = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    filenames, labels, logits, val_loss = evaluate(model, val_loader, val_criterion, device)
    p = probabilities(logits)
    metrics = compute_metrics(labels, p.argmax(1), p)
    if abs(metrics['macro_f1'] - best_score) > 1e-9:
        raise RuntimeError('Đánh giá lại checkpoint không khớp điểm val đã chọn')
    np.savez(out / 'val_outputs.npz', filenames=np.asarray(filenames), y_true=labels, logits=logits)
    save_predictions(pred_path(cfg, 'val'), filenames, labels, p)
    pred = read_pred(str(pred_path(cfg, 'val')))
    if not cfg.smoke_val:
        check_against_csv(pred, str(Path(cfg.labels_dir) / 'val_subset0.csv'), 'val')
    summary = {'exp_id': cfg.exp_id, 'seed': cfg.seed, 'config': asdict(cfg), 'model': meta,
               'best_epoch': best_epoch, 'val': metrics, 'val_loss': val_loss,
               'epochs_completed': len(history), 'seconds_total': sum(h['seconds'] for h in history),
               'seconds_per_epoch': float(np.mean([h['seconds'] for h in history])),
               'training_seconds_per_epoch': float(np.mean([h['train_seconds'] for h in history])),
               'best_checkpoint': str(out / 'best.pt'),
               'checkpoint_sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
               'curve': str(curve), 'train_class_counts': train_counts, 'test_evaluated': False}
    write_json(out / 'summary.json', summary)
    if cfg.save_test_predictions:
        from inference import export_frozen_test
        export_frozen_test(cfg)
        summary['test_evaluated'] = True
        write_json(out / 'summary.json', summary)
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    hints = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'Cần KEY=VALUE: {pair}')
        key, raw = pair.split('=', 1)
        if key not in hints or key in result:
            raise ValueError(f'Field không có hoặc bị lặp: {key}')
        hint = hints[key]
        options = get_args(hint) or (hint,)
        if raw.lower() in ('none', 'null'):
            if type(None) not in options:
                raise ValueError(f'{key} không cho phép None')
            value = None
        else:
            kind = next(t for t in options if t is not type(None))
            if kind is bool:
                if raw.lower() not in ('true','false'):
                    raise ValueError(f'{key} cần true/false')
                value = raw.lower() == 'true'
            else:
                value = kind(raw)
                if kind is float and not math.isfinite(value):
                    raise ValueError(f'{key} cần số hữu hạn')
        result[key] = value
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description='Huấn luyện DeepWeeds qua một engine')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), default=json_safe, ensure_ascii=False))


if __name__ == '__main__':
    main()
