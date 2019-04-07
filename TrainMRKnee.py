"""
Train a model for the MURA detection task.
"""
import os
import datetime
import argparse
from tqdm import tqdm
from functools import partial
import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.aug import resize_aug, simple_aug, pad_image
from utils.helper import LossTracker, preprocess_input
from data.loader import KneeDataset
from model import pretrained
from model.layer import FCLayer
from model.train import KneeTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help="Name of the model architecture.")
    parser.add_argument('save_dir', type=str, help="Directory to save the trained model.")
    parser.add_argument('data_dir', type=str, help="Directory to load image data from.")
    parser.add_argument('fold', type=int)
    parser.add_argument('val_fold', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--training-mode', type=str, default='fine-tune-all')
    parser.add_argument('--log-dampened', type=str, default='yes') 
    parser.add_argument('--labels-df', type=str, default='acl_df_splits.csv')
    parser.add_argument('--imsize_x', type=int, default=384)
    parser.add_argument('--imsize_y', type=int, default=384)
    parser.add_argument('--imratio', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--augment-p', type=float, default=0.5)
    parser.add_argument('--dropout-p', type=float, default=0.2)
    parser.add_argument('--max-epochs', type=int, default=100) 
    parser.add_argument('--steps-per-epoch', type=int, default=0)
    parser.add_argument('--head-max-epochs', type=int, default=5)
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6) 
    parser.add_argument('--lr-patience', type=int, default=2) 
    parser.add_argument('--stop-patience', type=int, default=10) 
    parser.add_argument('--annealing-factor', type=float, default=0.5)
    parser.add_argument('--min-delta', type=float, default=1e-3) 
    parser.add_argument('--verbosity', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--save_best', action='store_true', help='Only store the best model.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    train_aug = simple_aug(p=args.augment_p)
    resize_me = resize_aug(imsize_x=args.imsize_x, imsize_y=args.imsize_y)
    pad_func = partial(pad_image, ratio=args.imratio)
    
    print ("Training the MR KNEE ACL TEAR model...")
    
    if args.training_mode not in ['fine-tune-all', 'fc-only']: 
        raise Exception('training-mode must be one of : [fine-tune-all, fc-only]')
    
    torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    print("Saving model to {}".format(args.save_dir))
    
    print("Reading labels from {}".format(args.labels_df))
    df = pd.read_csv(args.labels_df)
    df['label'] = [1 if _ == 'ACL Tear' else 0 for _ in df['labelId']]
 
    df = df[df['fold'] != args.fold]
    train_df = df[df['val{}'.format(args.val_fold)] == 'train']
    valid_df = df[df['val{}'.format(args.val_fold)] == 'valid']
    
    print ('TRAIN: n={}'.format(len(train_df)))
    print ('VALID: n={}'.format(len(valid_df)))
    
    print("Reading images from directory {}".format(args.data_dir))
    train_images = [os.path.join(args.data_dir, _) for _ in train_df['filepath']]
    valid_images = [os.path.join(args.data_dir, _) for _ in valid_df['filepath']]
    train_labels = list(train_df['label'])
    valid_labels = list(valid_df['label'])
    num_classes = len(np.unique(train_labels)) 

    params = {'batch_size':  args.batch_size, 
              'shuffle':     True, 
              'num_workers': args.num_workers}
    
    valid_params = {'batch_size':  args.batch_size, 
                    'shuffle':     False, 
                    'num_workers': args.num_workers}    

    # Run model script 
    print ('Loading pretrained model [{}] ...'.format(args.model)) 
    model_func = getattr(pretrained, args.model)
    model, dim_feats = model_func()
    model.train().cuda()
    
    classifier = FCLayer(in_features=dim_feats, num_classes=num_classes, dropout=args.dropout_p)
    classifier = classifier.train().cuda() 
    _, weights = np.unique(train_labels, return_counts=True) 
    weights = weights / float(np.sum(weights))
    weights = 1. / weights
    if args.log_dampened == 'yes': 
        weights = 1 + np.log(weights)
    weights /= np.sum(weights)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).type('torch.FloatTensor')).cuda()
    
    if args.training_mode == 'fine-tune-all': 
        train_params = list(model.parameters()) + list(classifier.parameters())
    elif args.training_mode == 'fc-only': 
        train_params = list(classifier.parameters())
    
    optimizer = optim.Adam(train_params, 
                           lr=args.initial_lr,
                           weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                     factor=args.annealing_factor, 
                                                     patience=args.lr_patience, 
                                                     threshold=args.min_delta, 
                                                     threshold_mode='abs', 
                                                     verbose=True)
    
    # Set up preprocessing function with model 
    ppi = partial(preprocess_input, model=model) 
    
    print ('Setting up data loaders ...')
    
    # Create IDs to coalesce image-level predictions to study-level predictions 
    # during validation

    levels = valid_df['StudyInstanceUID']
    
    # Convert to integer
    levels_dict = {_ : i for i, _ in enumerate(np.unique(levels))}
    levels = [levels_dict[_] for _ in levels]

    train_set = KneeDataset(imgfiles=train_images,
                            labels=train_labels,
                            preprocess=ppi, 
                            transform=train_aug,
                            pad=pad_func,
                            resize=resize_me)
    train_gen = DataLoader(train_set, **params) 
    
    valid_set = KneeDataset(imgfiles=valid_images,
                            labels=valid_labels,
                            levels=levels,
                            preprocess=ppi, 
                            pad=pad_func,
                            resize=resize_me)
    valid_gen = DataLoader(valid_set, **valid_params) 
    
    loss_tracker = LossTracker() 
    
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch == 0: 
        steps_per_epoch = int(np.ceil(train_df.shape[0] / args.batch_size))
    
    trainer = KneeTrainer(model, args.model, optimizer, criterion, loss_tracker, args.save_dir, classifier, args.save_best)
    trainer.set_dataloaders(train_gen, valid_gen) 
    
    if args.training_mode == 'fine-tune-all':
        trainer.train_head(optim.Adam(classifier.parameters()), steps_per_epoch, args.head_max_epochs)
    
    trainer.train(args.max_epochs, steps_per_epoch, scheduler, args.stop_patience, verbosity=args.verbosity)

if __name__ == '__main__':
    main()
