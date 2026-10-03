import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from losses import FocalLoss, LabelSmoothingCE, class_weights, weights_from_train, mix_batch, mixed_loss, soft_targets


class LossTests(unittest.TestCase):
    def test_focal_zero_equals_ce_and_weighted_ce(self):
        torch.manual_seed(1)
        z=torch.randn(32,9,requires_grad=True); y=torch.arange(32)%9
        torch.testing.assert_close(FocalLoss(0)(z,y), F.cross_entropy(z,y),atol=1e-6,rtol=1e-5)
        weight=torch.arange(1,10,dtype=torch.float32)
        torch.testing.assert_close(FocalLoss(0,weight)(z,y), F.cross_entropy(z,y,weight=weight),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(LabelSmoothingCE(0)(z,y),F.cross_entropy(z,y))

    def test_mixup_and_cutmix_targets_area_and_loss(self):
        torch.manual_seed(4); np.random.seed(4)
        x=torch.stack([torch.full((3,32,32),float(i)) for i in range(9)])
        y=torch.arange(9); z=torch.randn(9,9)
        for mode in ['mixup','cutmix']:
            xm,targets=mix_batch(x,y,mode=mode)
            ya,yb,lam=targets
            self.assertEqual(xm.shape,x.shape); self.assertTrue(0<=lam<=1)
            q=soft_targets(targets)
            self.assertTrue((q>=0).all()); torch.testing.assert_close(q.sum(1),torch.ones(9))
            torch.testing.assert_close(mixed_loss(F.cross_entropy,z,targets),F.cross_entropy(z,q),atol=1e-6,rtol=1e-5)
            for i in range(9):
                if yb[i] != ya[i]:
                    if mode == 'cutmix':
                        actual=(xm[i,0]!=x[i,0]).float().mean().item()
                        self.assertAlmostEqual(actual,1-lam,places=7)
                    else:
                        torch.testing.assert_close(xm[i],lam*x[i]+(1-lam)*x[int(yb[i])])
            self.assertTrue(torch.equal(x[0],torch.zeros_like(x[0])))

    def test_training_weights_only_and_normalized(self):
        df=pd.DataFrame({'Label':sum(([i]*(i+1) for i in range(9)),[])})
        df.attrs.update(split='train',fold=0)
        a=weights_from_train(df)
        self.assertAlmostEqual(a.mean().item(),1.,places=6)
        torch.testing.assert_close(a,class_weights(np.arange(1,10)))
        b=class_weights(np.arange(1,10),beta=.9999)
        self.assertTrue(torch.isfinite(b).all())
        df.attrs['split']='val'
        with self.assertRaises(ValueError): weights_from_train(df)


if __name__ == '__main__': unittest.main()
