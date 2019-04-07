"""
Utilities for data augmentation.
"""
import cv2
import numpy as np

from albumentations import (
    Compose, OneOf, HorizontalFlip, ShiftScaleRotate, JpegCompression, Blur, CLAHE, RandomGamma, RandomContrast, RandomBrightness, Resize, PadIfNeeded
)


def simple_aug(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=10, scale_limit=0.15, p=0.5),
        OneOf([
            JpegCompression(quality_lower=80),
            Blur(),
        ], p=0.5),
        OneOf([
            CLAHE(),
            RandomGamma(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.5)
    ], p=p)

def resize_aug(imsize_x, imsize_y=None):
    if imsize_y is None: imsize_y = imsize_x
    return Compose([
        Resize(imsize_x, imsize_y, always_apply=True, p=1)
        ], p=1)

def pad_image(img, ratio=1.):
    # Default is ratio=1 aka pad to create square image
    ratio = float(ratio)
    # Given ratio, what should the height be given the width? 
    h, w = img.shape[:2]
    desired_h = int(w * ratio)
    # If the height should be greater than it is, then pad top/bottom
    if desired_h > h: 
        hdiff = int(desired_h - h)
        pad_list = [(hdiff / 2, desired_h-h-hdiff / 2), (0,0), (0,0)]
    # If height is smaller than it is, then pad left/right
    elif desired_h < h: 
        desired_w = int(h / ratio)
        wdiff = int(desired_w - w) 
        pad_list = [(0,0), (wdiff / 2, desired_w-w-wdiff / 2), (0,0)]
    elif desired_h == h: 
        return img 
    return np.pad(img, pad_list, 'constant', constant_values=np.min(img))
    

