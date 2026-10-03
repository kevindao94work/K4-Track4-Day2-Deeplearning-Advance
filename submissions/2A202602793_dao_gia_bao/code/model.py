"""Factory timm, đóng băng đúng chế độ và đo kích thước/tính toán tại chỗ."""
from __future__ import annotations

import copy
from types import MethodType

import timm
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

SUGGESTED_BACKBONES = {
    'resnet50': 'resnet50', 'resnext50': 'resnext50_32x4d',
    'convnext_tiny': 'convnext_tiny', 'deit_small': 'deit_small_patch16_224',
    'swin_tiny': 'swin_tiny_patch4_window7_224', 'efficientnet_b0': 'efficientnet_b0',
    'mobilenetv3': 'mobilenetv3_large_100',
}
# Đủ bốn họ bắt buộc, giảm chi phí trên Apple M4 16 GiB.
SCREENING_MODELS = [
    'resnet18.a1_in1k',
    'convnext_atto.d2_in1k',
    'deit_tiny_patch16_224.fb_in1k',
    'efficientnet_b0.ra_in1k',
    'mobilenetv3_small_100.lamb_in1k',
]


def classifier(model):
    head = model.get_classifier()
    if isinstance(head, str):
        head = model.get_submodule(head)
    if not isinstance(head, nn.Module):
        raise TypeError('Factory yêu cầu classifier là nn.Module')
    return head


def frozen_train(self, mode=True):
    # Giữ toàn bộ backbone ở eval kể cả khi gọi lại model.train().
    nn.Module.train(self, False)
    classifier(self).train(mode)
    self.training = mode
    return self


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = 'finetune', img_size: int = 224):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(f'Chế độ khởi tạo không hợp lệ: {init}')
    name = SUGGESTED_BACKBONES.get(name, name)
    kw = {}
    architecture = name.split('.')[0]
    if architecture.startswith(('deit', 'vit')):
        kw.update(img_size=img_size, dynamic_img_size=True)
    elif architecture.startswith('swin'):
        kw['img_size'] = img_size
    model = timm.create_model(name, pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate, **kw)
    model.lab_architecture = architecture
    cfg = model.pretrained_cfg
    model.lab_pretrained_tag = architecture + ('.' + cfg['tag'] if cfg.get('tag') else '')
    model.lab_pretrained_used = bool(pretrained and init != 'scratch')
    if init == 'frozen':
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    ids = {id(p) for p in classifier(model).parameters()}
    if not ids:
        raise ValueError('Không tìm thấy head có tham số')
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    model.train = MethodType(frozen_train, model)
    model.lab_frozen = True
    model.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = {id(p) for p in classifier(model).parameters()}
    norm_ids = set()
    norms = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
             nn.modules.instancenorm._InstanceNorm, nn.LocalResponseNorm)
    for module in model.modules():
        if isinstance(module, norms):
            norm_ids.update(id(p) for p in module.parameters(recurse=False))
    special = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = id(p) in head_ids
        no_decay = id(p) in norm_ids or name.endswith('.bias') or p.ndim <= 1 or name in special
        key = ('head' if head else 'backbone', no_decay)
        if key not in groups:
            groups[key] = {'params': [], 'param_names': [], 'lr': lr_head if head else lr_backbone,
                           'weight_decay': 0.0 if no_decay else weight_decay,
                           'group_name': key[0] + ('_no_decay' if no_decay else '_decay')}
        groups[key]['params'].append(p)
        groups[key]['param_names'].append(name)
    return list(groups.values())


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_gmacs(model, img_size: int = 224) -> float:
    # PyTorch đếm toán tử aten thực sự chạy; quy đổi FLOP/2 thành MAC.
    measured = copy.deepcopy(model).cpu().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        measured(x)
    flops = counter.get_total_flops()
    if flops <= 0:
        raise RuntimeError('Không đo được FLOP của backbone này')
    return float(flops / 2e9)


def model_metadata(model, img_size=224):
    return {'architecture': model.lab_architecture, 'pretrained_tag': model.lab_pretrained_tag,
            'pretrained_used': model.lab_pretrained_used,
            'pretrained_cfg': model.pretrained_cfg,
            'parameters_m': count_params(model), 'trainable_parameters': count_trainable(model),
            'gmac': count_gmacs(model, img_size),
            'gmac_tool': 'torch.utils.flop_counter.FlopCounterMode, FLOP/2; toán tử được hỗ trợ'}
