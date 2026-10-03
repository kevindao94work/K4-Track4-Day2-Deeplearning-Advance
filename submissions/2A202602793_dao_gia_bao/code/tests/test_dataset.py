"""Các lỗi đầu vào và tính xác định dễ làm sai trong pipeline."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (DeepWeedsDataset, build_transforms, denormalize, load_split,
                     make_loader, set_seed, validate_frame)


class DatasetTests(unittest.TestCase):
    def test_reject_invalid_labels_duplicates_paths(self):
        for df in [pd.DataFrame({'Filename':['x.jpg'], 'Label':[9]}),
                   pd.DataFrame({'Filename':['x.jpg'], 'Label':[.5]}),
                   pd.DataFrame({'Filename':['x.jpg','x.jpg'], 'Label':[0,1]}),
                   pd.DataFrame({'Filename':['../x.jpg'], 'Label':[0]})]:
            with self.assertRaises(ValueError):
                validate_frame(df, 'kiểm tra')

    def test_rgb_filename_and_deterministic_val(self):
        with tempfile.TemporaryDirectory() as d:
            Image.fromarray(np.full((256,256), 128, dtype=np.uint8)).save(Path(d)/'gray.png')
            df = pd.DataFrame({'Filename':['gray.png'], 'Label':[2]})
            ds = DeepWeedsDataset(df, d, build_transforms(False))
            a,y,name = ds[0]
            b,_,_ = ds[0]
            self.assertEqual((y,name), (2,'gray.png'))
            self.assertEqual(tuple(a.shape), (3,224,224))
            self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(denormalize(a), torch.full_like(a, 128/255))

    def test_cached_columns_preserve_source_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            for name,value in [('b.png',72),('a.png',160)]:
                Image.fromarray(np.full((16,16,3),value,dtype=np.uint8)).save(Path(d)/name)
            df=pd.DataFrame({'Filename':['b.png','a.png'],'Label':[7,0]},index=[20,10])
            df.attrs={'reference_names':{'b.png','a.png'},'split':'train'}
            original=df.copy(deep=True)
            ds=DeepWeedsDataset(df,d,build_transforms(False,16))
            self.assertEqual([ds[i][1:] for i in range(2)],[(7,'b.png'),(0,'a.png')])
            self.assertTrue(df.equals(original))
            self.assertEqual(df.attrs,original.attrs)
            df.loc[20,'Label']=1
            self.assertEqual(ds[0][1],7)

    def test_seeded_training_loader(self):
        with tempfile.TemporaryDirectory() as d:
            names = [f'{i}.png' for i in range(8)]
            image = np.arange(256*256*3, dtype=np.uint32).reshape(256,256,3).astype(np.uint8)
            for name in names:
                Image.fromarray(image).save(Path(d)/name)
            df = pd.DataFrame({'Filename':names, 'Label':np.arange(8)})
            set_seed(42)
            a = next(iter(make_loader(df,d,build_transforms(True),4,True,num_workers=0,seed=42)))
            set_seed(42)
            b = next(iter(make_loader(df,d,build_transforms(True),4,True,num_workers=0,seed=42)))
            self.assertEqual(a[2],b[2])
            self.assertTrue(torch.equal(a[0],b[0]))

    def test_fold_rejected_before_io(self):
        with self.assertRaises(ValueError):
            load_split('khong_ton_tai', fold=1)


if __name__ == '__main__':
    unittest.main()
