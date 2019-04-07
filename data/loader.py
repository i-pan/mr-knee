"""
Loaders for different datasets.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from utils.helper import channels_last_to_first, get_image_from_dicom

class GenericDataset(Dataset): 
    """
    Basic loader.
    """
    def __init__(self, imgfiles, labels, dicom=True, grayscale=True, preprocess=None, pad=None, resize=None, transform=None): 
        self.imgfiles   = imgfiles
        self.labels     = labels 
        self.dicom      = dicom
        self.grayscale  = grayscale
        self.preprocess = preprocess
        self.pad        = pad
        self.resize     = resize
        self.transform  = transform

    def __len__(self): 
        return len(self.imgfiles) 

    def __getitem__(self, i): 
        """
        Returns: x, y
            - x: tensorized input
            - y: tensorized label
        """
        # 1- Load image
        if self.dicom: 
            X = get_image_from_dicom(self.imgfiles[i])
        else:
            if self.grayscale: 
                mode = cv2.IMREAD_GRAYSCALE
            else: 
                mode = cv2.IMREAD_COLOR
            X = cv2.imread(self.imgfiles[i], mode)
        while X is None: 
            i = np.random.choice(len(self.imgfiles))
            if self.dicom: 
                X = get_image_from_dicom(self.imgfiles[i])
            else:
                X = cv2.imread(self.imgfiles[i], mode)
        if self.grayscale: 
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        # 2- Pad and resize image
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        # 3- Apply data augmentation
        if self.transform: X = self.transform(image=X)['image']
        # 4- Apply preprocessing
        if self.preprocess: X = self.preprocess(X)
        X = channels_last_to_first(X)
        y = np.asarray(self.labels[i])
        return torch.from_numpy(X).type('torch.FloatTensor'), \
               torch.from_numpy(y)

class KneeDataset(Dataset): 
    """
    Basic loader for MR KNEE dataset.
    """
    def __init__(self, imgfiles, labels, levels=None, dicom=True, grayscale=True, preprocess=None, pad=None, resize=None, transform=None): 
        self.imgfiles   = imgfiles
        self.labels     = labels 
        self.levels     = levels
        self.dicom      = dicom
        self.grayscale  = grayscale
        self.preprocess = preprocess
        self.pad        = pad
        self.resize     = resize
        self.transform  = transform

    def __len__(self): 
        return len(self.imgfiles) 

    def __getitem__(self, i): 
        """
        Returns: x, y, level
            - x: tensorized input data;
            - y: tensor of female binary indicators;
            - level [optional]: tensor of study level for coalescing predictions
        """
        # 1- Load image
        if self.dicom: 
            X = get_image_from_dicom(self.imgfiles[i])
        else:
            if self.grayscale: 
                mode = cv2.IMREAD_GRAYSCALE
            else: 
                mode = cv2.IMREAD_COLOR
            X = cv2.imread(self.imgfiles[i], mode)
        while X is None: 
            i = np.random.choice(len(self.imgfiles))
            if self.dicom: 
                X = get_image_from_dicom(self.imgfiles[i])
            else:
                X = cv2.imread(self.imgfiles[i], mode)
        if self.grayscale: 
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        # 2- Pad and resize image
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        # 3- Apply data augmentation
        if self.transform: X = self.transform(image=X)['image']
        # 4- Apply preprocessing
        if self.preprocess: X = self.preprocess(X)
        X = channels_last_to_first(X)
        y = np.asarray(self.labels[i])  
        if self.levels is not None: 
            level = np.asarray(self.levels[i])      
            return torch.from_numpy(X).type('torch.FloatTensor'), \
                   torch.from_numpy(y).type('torch.LongTensor'), \
                   torch.from_numpy(level).type('torch.FloatTensor') 
        else: 
            return torch.from_numpy(X).type('torch.FloatTensor'), \
                   torch.from_numpy(y).type('torch.LongTensor')

