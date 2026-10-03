import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import freeze_backbone, param_groups, build_model, classifier


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3,4,3,padding=1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4,9)

    def get_classifier(self):
        return self.head

    def forward(self,x):
        return self.head(self.backbone(x).flatten(1))


class ModelTests(unittest.TestCase):
    def test_new_head_has_controlled_small_initialization(self):
        m=build_model('mobilenetv3_small_100.lamb_in1k',pretrained=False)
        head=classifier(m)
        self.assertLess(abs(head.weight.std().item()-.01),.001)
        self.assertTrue(torch.equal(head.bias,torch.zeros_like(head.bias)))

    def test_freeze_preserves_parameters_and_bn_buffers_after_step(self):
        m=Toy()
        freeze_backbone(m)
        before={k:v.clone() for k,v in m.state_dict().items()}
        opt=torch.optim.AdamW(param_groups(m,1e-4,1e-3,.05))
        m.train()
        nn.functional.cross_entropy(m(torch.randn(4,3,8,8)), torch.tensor([0,1,2,3])).backward()
        opt.step()
        for k,v in m.state_dict().items():
            if k.startswith('backbone.'):
                self.assertTrue(torch.equal(v,before[k]), k)
        self.assertFalse(torch.equal(m.head.weight,before['head.weight']))

    def test_group_membership_lr_and_decay(self):
        m=Toy()
        seen=set()
        for g in param_groups(m,1e-4,1e-3,.05):
            for name,p in zip(g['param_names'],g['params']):
                self.assertNotIn(id(p),seen); seen.add(id(p))
                self.assertEqual(g['lr'], 1e-3 if name.startswith('head.') else 1e-4)
                self.assertEqual(g['weight_decay'], 0 if p.ndim <= 1 else .05)
        self.assertEqual(len(seen),len(list(m.parameters())))


if __name__ == '__main__':
    unittest.main()
