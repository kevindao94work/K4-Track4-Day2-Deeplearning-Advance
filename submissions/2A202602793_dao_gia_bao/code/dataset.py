"""Đọc DeepWeeds nguyên bản, kiểm tra fold và tạo DataLoader tái lập."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia',
               'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negatives']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_frame(df, source):
    if not {'Filename', 'Label'} <= set(df.columns) or len(df) == 0:
        raise ValueError(f'{source}: cần CSV không rỗng với Filename, Label')
    if df[['Filename', 'Label'] + (['Species'] if 'Species' in df else [])].isna().any().any():
        raise ValueError(f'{source}: dữ liệu thiếu')
    if df.Filename.duplicated().any():
        raise ValueError(f'{source}: tên ảnh trùng')
    labels = pd.to_numeric(df.Label, errors='raise').to_numpy()
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f'{source}: nhãn phải nguyên')
    if not np.isin(labels, np.arange(NUM_CLASSES)).all():
        raise ValueError(f'{source}: nhãn ngoài [0,8]')
    for name in df.Filename:
        p = Path(str(name))
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 1:
            raise ValueError(f'{source}: tên ảnh không an toàn: {name}')


def load_split(labels_dir: str | Path, fold: int = 0):
    if fold != 0:
        raise ValueError('Các thí nghiệm bắt buộc chỉ dùng fold 0')
    root = Path(labels_dir)
    ref_path = root / 'labels.csv'
    ref = pd.read_csv(ref_path)
    validate_frame(ref, ref_path)
    if 'Species' not in ref:
        raise ValueError('labels.csv cần cột Species')
    classes = ref[['Label', 'Species']].drop_duplicates().sort_values('Label')
    if classes.Label.tolist() != list(range(NUM_CLASSES)):
        raise ValueError('labels.csv phải ánh xạ duy nhất 9 lớp theo Label 0..8')
    mapping = dict(zip(ref.Filename, zip(ref.Label.astype(int), ref.Species)))
    frames = []
    hashes = {'labels.csv': hashlib.sha256(ref_path.read_bytes()).hexdigest()}
    for split in ('train', 'val', 'test'):
        path = root / f'{split}_subset{fold}.csv'
        df = pd.read_csv(path)
        validate_frame(df, path)
        conflicts = []
        for row in df.itertuples():
            expected = mapping.get(row.Filename)
            if expected is None:
                raise ValueError(f'{path}: ảnh không có trong labels.csv: {row.Filename}')
            if expected[0] != int(row.Label):
                conflicts.append({'Filename': row.Filename, 'split_label': int(row.Label), 'labels_csv_label': expected[0]})
            if 'Species' in df and row.Species != classes.Species.iloc[int(row.Label)]:
                raise ValueError(f'{path}: ánh xạ Label/Species không khớp labels.csv')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        df.attrs.update(fold=fold, split=split, source=str(path),
                        class_names=classes.Species.tolist(), reference_names=set(ref.Filename),
                        annotation_conflicts=conflicts)
        frames.append(df)
    for df in frames:
        df.attrs['csv_sha256'] = hashes
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    frames = dict(train=train_df, val=val_df, test=test_df)
    root = Path(images_dir)
    sets = {}
    for split, df in frames.items():
        validate_frame(df, split)
        if df.attrs.get('fold') != 0 or df.attrs.get('split') != split:
            raise ValueError('Phải kiểm tra split nguyên bản do load_split(fold=0) trả về')
        if not df.equals(pd.read_csv(df.attrs['source'])):
            raise ValueError(f'{split}: DataFrame đã bị sửa/lọc so với CSV nguyên bản')
        sets[split] = set(df.Filename)
    overlap = {f'{a}_{b}': len(sets[a] & sets[b])
               for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlap.values()):
        raise ValueError(f'Rò rỉ giữa các split: {overlap}')
    union = set.union(*sets.values())
    if len(union) != 17509 or union != train_df.attrs['reference_names']:
        raise ValueError(f'Hợp split phải khớp đúng 17.509 ảnh labels.csv, nhận {len(union)}')
    missing = [f for f in sorted(union) if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f'Thiếu {len(missing)} ảnh, ví dụ: {missing[:5]}')
    result = {'fold': 0, 'n': {s: len(df) for s, df in frames.items()},
              'per_class': {s: df.Label.value_counts().reindex(range(9), fill_value=0).astype(int).tolist()
                            for s, df in frames.items()},
              'class_names': train_df.attrs['class_names'], 'overlap': overlap,
              'union': len(union), 'missing': len(missing), 'num_classes': 9,
              'annotation_conflicts': {s: df.attrs['annotation_conflicts'] for s, df in frames.items()},
              'csv_sha256': train_df.attrs['csv_sha256']}
    fractions = np.array(list(result['n'].values())) / len(union)
    if (np.abs(fractions - np.array([.6, .2, .2])) > .01).any():
        raise ValueError(f'Tỉ lệ split lệch quá 1 điểm phần trăm: {fractions}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_transforms(train: bool, img_size: int = 224, aug: str = 'basic',
                     mean=IMAGENET_MEAN, std=IMAGENET_STD,
                     interpolation='bicubic', crop_pct=0.875):
    interp = InterpolationMode.BICUBIC if interpolation == 'bicubic' else InterpolationMode.BILINEAR
    if train:
        ops = [T.RandomResizedCrop(img_size, interpolation=interp), T.RandomHorizontalFlip()]
        if aug == 'color':
            ops += [T.ColorJitter(.3, .3, .3, .05)]
        elif aug == 'trivial':
            ops += [T.TrivialAugmentWide(interpolation=interp)]
        elif aug == 'randaug':
            ops += [T.RandAugment(interpolation=interp)]
        elif aug != 'basic':
            raise ValueError(f'Augmentation chưa hỗ trợ: {aug}')
    else:
        ops = [T.Resize(int(img_size / crop_pct), interpolation=interp), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    shape = (3, 1, 1) if x.ndim == 3 else (1, 3, 1, 1)
    return (x * x.new_tensor(std).reshape(shape) + x.new_tensor(mean).reshape(shape)).clamp(0, 1)


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        validate_frame(df, 'Dataset')
        self.df = df.copy(deep=True).reset_index(drop=True)
        # pandas sao chép attrs khi tạo Series qua iloc; metadata split lớn không
        # cần lặp lại cho mỗi ảnh. Cache cột, giữ nguyên thứ tự và nhãn nguồn.
        self.filenames = self.df.Filename.astype(str).to_numpy(copy=True)
        self.targets = self.df.Label.to_numpy(dtype=np.int64, copy=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = str(self.filenames[i])
        with Image.open(self.images_dir / filename) as im:
            image = im.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[i]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    ds = DeepWeedsDataset(df, images_dir, transform)
    if sampler not in (None, 'balanced') or (sampler and not train):
        raise ValueError('Sampler phải None hoặc balanced chỉ trên train')
    sampling = None
    if sampler == 'balanced':
        counts = df.Label.value_counts()
        weights = torch.as_tensor(df.Label.map(lambda k: 1 / counts[k]).to_numpy(), dtype=torch.double)
        sampling = WeightedRandomSampler(weights, len(ds), replacement=True, generator=generator)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sampling is None,
                      sampler=sampling, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(ds) >= batch_size,
                      worker_init_fn=seed_worker, generator=generator)
