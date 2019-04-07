"""
Access to pretrainded models.
"""
import pretrainedmodels
import pretrainedmodels.utils
import torch
from torch import nn

def densenet():
    """ Make densenet121 a default densenet. """
    return densenet121()

def densenet121():
    model = getattr(pretrainedmodels, 'densenet121')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def densenet161():
    model = getattr(pretrainedmodels, 'densenet161')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnet():
    """ Make resnet101 a default resnet. """
    return resnet101()

def resnet50():
    model = getattr(pretrainedmodels, 'resnet50')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnet101():
    model = getattr(pretrainedmodels, 'resnet101')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnet152():
    model = getattr(pretrainedmodels, 'resnet152')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def inceptionv3():
    model = getattr(pretrainedmodels, 'inceptionv3')(num_classes=1000, pretrained='imagenet')
    dim_feats = model.last_linear.in_features
    model.aux_logits = False
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def inceptionv4():
    model = getattr(pretrainedmodels, 'inceptionv4')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity() 
    return model, dim_feats

def se_resnext():
    return se_resnext50_32x4d()

def se_resnext50_32x4d():
    model = getattr(pretrainedmodels, 'se_resnext50_32x4d')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features 
    model.avg_pool = nn.AdaptiveAvgPool2d(1) 
    model.last_linear = pretrainedmodels.utils.Identity() 
    return model, dim_feats

def xception():
    model = getattr(pretrainedmodels, 'xception')(num_classes=1000, pretrained='imagenet') 
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats
