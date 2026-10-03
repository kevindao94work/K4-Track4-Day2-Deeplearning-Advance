"""Xuất/cache val không ghi đè bản dùng chọn checkpoint hoặc khớp T trên test."""
import sys,json,tempfile
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import final_training
from train import Config
from inference import Method,apply_temperature
from eval import save_predictions

class FinalValTests(unittest.TestCase):
    def test_validation_cache_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'predictions').mkdir();(root/'labels').mkdir()
            frozen_path=root/'freeze.json';frozen_path.write_text('{}')
            cfg=Config(exp_id='T00',seed=0,img_size=8,out_dir=str(root/'logs'))
            folder=root/'logs/T00/seed0';folder.mkdir(parents=True)
            names=np.array(['a.jpg','b.jpg','c.jpg']);y=np.array([0,7,8]);z=np.eye(9)[y]*3
            pd.DataFrame({'Filename':names,'Label':y}).to_csv(root/'labels/val_subset0.csv',index=False)
            np.savez(folder/'val_outputs.npz',filenames=names,y_true=y,logits=z)
            source=root/'predictions/T00_seed0_val.csv';save_predictions(source,names,y,apply_temperature(z,1))
            original=source.read_bytes();summary={'checkpoint_sha256':'fixture'}
            frozen={'configs':{'T00':{'inference':asdict(Method(resolution=8,input_size=8)), 'temperature_policy':'none'}}}
            with patch.multiple(final_training,SUB=root,DATA=root),patch.object(final_training,'frozen_path',return_value=frozen_path):
                r=final_training.final_val(cfg,summary,frozen)
                self.assertEqual(r['temperature'],1);self.assertIsNone(r['temperature_fit_split'])
                self.assertEqual((root/'predictions/T00_train_seed0_val.csv').read_bytes(),original)
                with patch.object(final_training,'predict_method',side_effect=AssertionError('Không được chạy lại')):
                    self.assertEqual(final_training.final_val(cfg,summary,frozen)['prediction_sha256'],r['prediction_sha256'])
                source.write_text('changed')
                with self.assertRaises(ValueError):final_training.final_val(cfg,summary,frozen)

if __name__=='__main__':unittest.main()
