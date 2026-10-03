"""Kiểm tra phép gộp, hiệu chỉnh, BN và quy trình đo độ trễ."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference import (aggregate_views,apply_temperature,fit_temperature,ensemble_probs,
                       view_hflip,views_multicrop,views_multiscale,fuse_conv_bn,predict_logits)
from benchmark import bench,latency_report,tta_latency


class InferenceTests(unittest.TestCase):
    def test_views(self):
        x=torch.arange(2*3*8*8,dtype=torch.float32).reshape(2,3,8,8)
        self.assertTrue(torch.equal(view_hflip(view_hflip(x)),x))
        crops=views_multicrop(x,4)
        self.assertEqual(len(crops),5);self.assertTrue(torch.equal(crops[-1],x[...,2:6,2:6]))
        self.assertEqual([tuple(v.shape[-2:]) for v in views_multiscale(x,[4,12])],[(4,4),(12,12)])
        with self.assertRaises(ValueError):views_multicrop(x,9)

    def test_aggregation_and_ensemble(self):
        a=np.zeros((3,9));b=a.copy();a[:,0]=6;b[:,1]=1
        prob=aggregate_views([a,b],'prob');logit=aggregate_views([a,b],'logit')
        np.testing.assert_allclose(prob,(apply_temperature(a,1)+apply_temperature(b,1))/2)
        np.testing.assert_allclose(logit,apply_temperature((a+b)/2,1))
        self.assertFalse(np.allclose(prob,logit));np.testing.assert_allclose(prob.sum(1),1)
        with self.assertRaises(ValueError):ensemble_probs([a,b])
        with self.assertRaises(ValueError):aggregate_views([a,b[:1]],'prob')

    def test_temperature_nll_and_argmax(self):
        rng=np.random.default_rng(12);z=rng.normal(size=(100,9))*6;y=rng.integers(0,9,100)
        t=fit_temperature(z,y);p=apply_temperature(z,t);raw=apply_temperature(z,1)
        self.assertGreater(t,0);np.testing.assert_array_equal(raw.argmax(1),p.argmax(1))
        self.assertLessEqual(-np.log(p[np.arange(100),y]).mean(),-np.log(raw[np.arange(100),y]).mean()+1e-9)
        with self.assertRaises(ValueError):apply_temperature(z,0)

    def test_bn_fusion_copy(self):
        torch.manual_seed(4)
        m=nn.Sequential(nn.Conv2d(3,6,3,padding=1,bias=False),nn.BatchNorm2d(6),nn.ReLU()).eval()
        x=torch.randn(2,3,10,10);f=fuse_conv_bn(m)
        self.assertIsInstance(m[1],nn.BatchNorm2d);self.assertIsInstance(f[1],nn.Identity)
        self.assertEqual(f.lab_fused_pairs,1)
        with torch.inference_mode():torch.testing.assert_close(m(x),f(x),atol=1e-5,rtol=1e-5)

    def test_predict_order_eval(self):
        m=nn.Sequential(nn.Flatten(),nn.Dropout(.9),nn.Linear(12,9)).train()
        loader=[(torch.randn(2,3,2,2),torch.tensor([7,0]),['b','a'])]
        names,y,z=predict_logits(m,loader,'cpu');self.assertFalse(m.training)
        self.assertEqual(names,['b','a']);np.testing.assert_array_equal(y,[7,0]);self.assertEqual(z.shape,(2,9))
        np.testing.assert_array_equal(z,predict_logits(m,loader,'cpu')[2])

    def test_benchmark_sync_and_counts(self):
        calls={'forward':0,'sync':0}
        def fn():calls['forward']+=1
        def sync():calls['sync']+=1
        r=bench(fn,10,50,sync)
        self.assertEqual(calls,{'forward':60,'sync':101});self.assertEqual(r['n'],50)
        self.assertLessEqual(r['p50'],r['p95']);self.assertLessEqual(r['p95'],r['p99'])
        with self.assertRaises(ValueError):bench(fn,0,5)
        m=nn.Sequential(nn.Conv2d(3,9,1),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        r=latency_report(m,2,8,device='cpu',iters=50)
        self.assertEqual(r['batch'],2);self.assertGreater(r['images_per_s'],0)
        self.assertEqual(tta_latency(m,2,device='cpu',img_size=8,iters=50)['K'],2)


if __name__=='__main__':unittest.main()
