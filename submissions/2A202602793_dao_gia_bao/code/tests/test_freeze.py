"""Cổng chốt từ chối seed/công thức/hash đã thay đổi."""
import sys,json,tempfile,hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import freeze_final
from train import Config,frozen_recipe,validate_freeze

class FreezeTests(unittest.TestCase):
    def test_recipe_seed_id_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'freeze.json';cfg=Config(exp_id='F01',seed=1,freeze_file=str(p))
            p.write_text(json.dumps({'seeds':[0,1,2],'configs':{'F01':{'recipe':frozen_recipe(cfg)}}}))
            validate_freeze(cfg)
            for bad in [replace(cfg,seed=3),replace(cfg,exp_id='OTHER'),replace(cfg,lr_head=.02)]:
                with self.assertRaises(ValueError):validate_freeze(bad)

    def test_integrity_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'labels').mkdir()
            for p in [root/'model.py',root/'eval.py',root/'labels/labels.csv']:p.write_text('fixture')
            digest=hashlib.sha256(b'fixture').hexdigest()
            r={'inference_fingerprint':freeze_final.inference_fingerprint(),
               'pipeline_sha256':{'model.py':digest},'csv_sha256':{'labels.csv':digest},'evaluator_sha256':digest}
            with patch.multiple(freeze_final,CODE=root,DATA=root,REPO=root):
                freeze_final.assert_integrity(r)
                (root/'labels/labels.csv').write_text('changed')
                with self.assertRaises(ValueError):freeze_final.assert_integrity(r)

if __name__=='__main__':unittest.main()
