"""Tải nguồn chính thức theo các đoạn có thể tiếp tục; kiểm MD5 trước giải nén."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/7939060/files/images.zip?download=1'
MD5 = 'b7b30f96d466fba86016aa5a26606e0f'
TOTAL = 491516047
BASE = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/'


def download(root, workers=8):
    root = Path(root)
    (root / 'labels').mkdir(parents=True, exist_ok=True)
    for name in ['labels', 'train_subset0', 'val_subset0', 'test_subset0']:
        path = root / 'labels' / (name + '.csv')
        if not path.exists():
            urllib.request.urlretrieve(BASE + name + '.csv', path)
    target = root / 'images.zip'
    if not target.exists() or target.stat().st_size != TOTAL:
        parts = root / 'download_parts'
        parts.mkdir(exist_ok=True)
        size = (TOTAL + workers - 1) // workers
        # Tiếp tục phần đầu đã tải bởi một kết nối trước đó.
        if target.exists() and not (parts / 'part0').exists():
            with target.open('rb') as src, (parts / 'part0').open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1 << 20) if target.stat().st_size < size else dst.write(src.read(size))

        def fetch(i):
            start, end = i * size, min((i + 1) * size, TOTAL) - 1
            path = parts / f'part{i}'
            expected = end - start + 1
            for attempt in range(5):
                offset = path.stat().st_size if path.exists() else 0
                if offset == expected:
                    return path
                try:
                    req = urllib.request.Request(URL, headers={'Range': f'bytes={start + offset}-{end}'})
                    with urllib.request.urlopen(req, timeout=60) as src:
                        if src.status != 206 or src.headers.get('Content-Range') != f'bytes {start + offset}-{end}/{TOTAL}':
                            raise ValueError('Máy chủ không trả đúng đoạn yêu cầu')
                        with path.open('ab') as dst:
                            shutil.copyfileobj(src, dst, length=1 << 18)
                    if path.stat().st_size != expected:
                        raise ValueError('Đoạn tải thiếu byte')
                    print(f'Đã tải đoạn {i+1}/{workers}', flush=True)
                    return path
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(fetch, range(workers)))
        tmp = root / 'images.complete.zip'
        with tmp.open('wb') as dst:
            for path in paths:
                with path.open('rb') as src:
                    shutil.copyfileobj(src, dst)
        tmp.replace(target)
    h = hashlib.md5()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != MD5:
        raise ValueError(f'MD5 sai: {h.hexdigest()}')
    print('MD5 hợp lệ:', h.hexdigest(), flush=True)
    images = root / 'images'
    images.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            if not (images / info.filename).resolve().is_relative_to(images.resolve()):
                raise ValueError('Đường dẫn ZIP không an toàn')
        archive.extractall(images)
    print('Giải nén hoàn tất', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    download(args.data_dir, args.workers)
