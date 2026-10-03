"""Ledger độc quyền và xác minh cache không cần forward model."""
import sys,tempfile,hashlib
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_once import begin_pass,validate_cached
from eval import save_predictions

class TestLedgerTests(unittest.TestCase):
    def test_once_and_completed_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'labels').mkdir();(root/'predictions').mkdir()
            cfg=SimpleNamespace(labels_dir=str(root/'labels'),pred_dir=str(root/'predictions'))
            names=['a','b','c'];y=np.array([0,7,8]);p=np.eye(9)[y]
            pd.DataFrame({'Filename':names,'Label':y}).to_csv(root/'labels/test_subset0.csv',index=False)
            path=root/'predictions/F01_seed0_test.csv';save_predictions(path,names,y,p)
            ledger=root/'ledger.json';r=begin_pass(ledger,{'n':3,'frozen_sha256':'freeze','checkpoint_sha256':'checkpoint'})
            with self.assertRaises(ValueError):begin_pass(ledger,{})
            with self.assertRaises(ValueError):validate_cached(r,cfg,'freeze','checkpoint')
            r.update(status='complete',prediction_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()})
            self.assertEqual(validate_cached(r,cfg,'freeze','checkpoint')['pass_count'],1)
            with self.assertRaises(ValueError):validate_cached(r,cfg,'changed','checkpoint')
            path.write_text('changed')
            with self.assertRaises(ValueError):validate_cached(r,cfg,'freeze','checkpoint')

if __name__=='__main__':unittest.main()
