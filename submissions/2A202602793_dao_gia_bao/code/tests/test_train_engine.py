import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import Config, EMA, build_scheduler, parse_overrides, validate_freeze


class EngineTests(unittest.TestCase):
    def test_scheduler_warmup_cosine_and_ratio(self):
        p=nn.Parameter(torch.ones(1)); q=nn.Parameter(torch.ones(1))
        opt=torch.optim.SGD([{'params':[p],'lr':1e-4},{'params':[q],'lr':1e-3}])
        sch=build_scheduler(opt,Config(epochs=2,warmup_epochs=1),4)
        lr=[opt.param_groups[0]['lr']]
        for _ in range(8):
            opt.step(); sch.step(); lr.append(opt.param_groups[0]['lr'])
            if opt.param_groups[0]['lr']>0:
                self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],10)
        self.assertGreater(lr[3],lr[0]); self.assertEqual(lr[-1],0)
        self.assertTrue(all(a>=b for a,b in zip(lr[4:],lr[5:])))

    def test_ema_averages_parameters_copies_integer_buffers(self):
        m=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2))
        with torch.no_grad(): m[0].weight.fill_(1)
        ema=EMA(m,.5)
        with torch.no_grad():
            m[0].weight.fill_(3); m[1].num_batches_tracked.fill_(7)
        ema.update(m)
        torch.testing.assert_close(ema.model[0].weight,torch.full_like(m[0].weight,2))
        self.assertEqual(ema.model[1].num_batches_tracked.item(),7)
        self.assertFalse(ema.model.training)

    def test_overrides_and_sealed_test(self):
        self.assertEqual(parse_overrides(['seed=2','amp=false','ema_decay=none','lr_head=1e-3']),
                         {'seed':2,'amp':False,'ema_decay':None,'lr_head':.001})
        for items in [['unknown=1'],['amp=yes'],['seed=none'],['seed=1','seed=2']]:
            with self.assertRaises(ValueError): parse_overrides(items)
        with self.assertRaises(ValueError): validate_freeze(Config(save_test_predictions=True))


if __name__ == '__main__': unittest.main()
