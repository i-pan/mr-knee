"""
Objects for training models.
"""

import os
import datetime
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score
from functools import partial

from utils.helper import to_categorical

def _roc_auc_score(y_true, y_pred):
    y_true = np.asarray(y_true) 
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred[:,1])
    else:
        auc = roc_auc_score(to_categorical(y_true), y_pred, average='macro') 
    return auc

def _f1_score(y_true, y_pred):
    y_true = np.asarray(y_true) 
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1))
    else:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')     
    return f1

class Trainer(object): 
    def __init__(self, model, architecture, optimizer, criterion, loss_tracker, save_checkpoint, classifier, save_best):
        self.model = model 
        self.architecture = architecture 
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.classifier = classifier 
        self.save_checkpoint = save_checkpoint
        self.save_best = save_best
        self.best_model = None
 
    def set_dataloaders(self, train, valid=None): 
        self.train_gen = train 
        self.valid_gen = valid

    def check_end_train(self): 
        return True if self.current_epoch >= self.max_epochs else False

    def train_head(self, head_optimizer, head_steps_per_epoch, head_max_epochs=5): 
        print ('Training head for {} epochs ...'.format(head_max_epochs))
        head_current_epoch = 0 ; head_steps = 0
        while True: 
            for i, data in enumerate(self.train_gen):
                batch, labels = data  
                head_optimizer.zero_grad()
                output = self.model(batch.cuda())
                output = self.classifier(output.cuda())
                loss = self.criterion(output, labels.cuda())
                loss.backward() 
                head_optimizer.step()
                head_steps += 1
                if head_steps % head_steps_per_epoch == 0: 
                    head_current_epoch += 1
                    head_steps = 0
                    if head_current_epoch >= head_max_epochs: 
                        break
            if head_current_epoch >= head_max_epochs: 
                break
        print ('Done training head !')

    def save_models(self, improvement, metrics):
        cpt_name = '{arch}_{epoch}'.format(arch=self.architecture.upper(), epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))))
        for met in metrics.keys(): 
            cpt_name += '_{name}-{value:.4f}'.format(name=met.upper(), value=metrics[met])
        cpt_name += '.pth'
        if not self.save_best: 
            torch.save(self.model.state_dict(), os.path.join(self.save_checkpoint, cpt_name))
            torch.save(self.classifier.state_dict(), os.path.join(self.save_checkpoint, 'HEAD_{}'.format(cpt_name)))
        elif improvement:
            if self.best_model is not None: 
                os.system('rm {}'.format(os.path.join(self.save_checkpoint, self.best_model)))
                os.system('rm {}'.format(os.path.join(self.save_checkpoint, 'HEAD_{}'.format(self.best_model))))
            self.best_model = cpt_name
            torch.save(self.model.state_dict(), os.path.join(self.save_checkpoint, cpt_name))
            torch.save(self.classifier.state_dict(), os.path.join(self.save_checkpoint, 'HEAD_{}'.format(cpt_name)))

    def calculate_valid_metrics(self, y_true, y_pred, loss): 
        valid_auc = _roc_auc_score(y_true, y_pred)
        valid_f1  = _f1_score(y_true, y_pred)
        print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, auc = {auc:.4f}, f1 = {f1:.4f}'
               .format(epoch=str(self.current_epoch + 1).zfill(len(str(self.max_epochs))), \
                       loss=loss, \
                       auc=valid_auc, \
                       f1=valid_f1))
        valid_metric = valid_auc
        metrics_dict = {'auc': valid_auc, 'f1': valid_f1}
        return valid_metric, metrics_dict

    def post_validate(self, valid_metric, metrics_dict):
        self.lr_scheduler.step(valid_metric)
        if self.lr_scheduler.mode == 'min': 
            improvement = valid_metric <= (self.best_valid_score - self.lr_scheduler.threshold)
        else: 
            improvement = valid_metric >= (self.best_valid_score + self.lr_scheduler.threshold) 
        self.save_models(improvement, metrics_dict)
        if improvement: 
            self.best_valid_score = valid_metric 
            self.stopping = 0 
        else: 
            self.stopping += 1

    def validate(self): 
        with torch.no_grad():
            self.model = self.model.eval().cuda()
            self.classifier = self.classifier.eval().cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                batch, labels = data  
                output = self.model(batch.cuda())
                output = self.classifier(output.cuda())
                loss = self.criterion(output, labels.cuda())
                y_pred.append(output.cpu().numpy())
                y_true.extend(labels.numpy())
                valid_loss += loss.item()
        y_pred = np.vstack(y_pred) 
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, valid_loss)
        self.post_validate(valid_metric, metrics_dict)

    def train_step(self, data): 
        batch, labels = data  
        self.optimizer.zero_grad()
        output = self.model(batch.cuda())
        output = self.classifier(output.cuda())
        loss = self.criterion(output, labels.cuda())
        self.loss_tracker.update_loss(loss.item()) 
        loss.backward() 
        self.optimizer.step()

    def train(self, max_epochs, steps_per_epoch, lr_scheduler=None, early_stopping=np.inf, verbosity=100): 
        self.lr_scheduler = lr_scheduler 
        self.best_valid_score = 999. if lr_scheduler.mode == 'min' else 0.
        self.max_epochs = max_epochs
        self.stopping = 0
        start_time = datetime.datetime.now() ; steps = 0 
        print ('TRAINING : START')
        self.current_epoch = 0
        while True: 
            for i, data in enumerate(self.train_gen):
                step_start_time = time.time()
                self.train_step(data)
                steps += 1
                if steps % verbosity == 0:
                    duration = time.time() - step_start_time
                    print('epoch {epoch}, batch {batch} / {steps_per_epoch} : loss = {train_loss:.4f} ({duration:.3f} sec/batch)'
                            .format(epoch=str(self.current_epoch + 1).zfill(len(str(max_epochs))), \
                                    batch=str(steps).zfill(len(str(steps_per_epoch))), \
                                    steps_per_epoch=steps_per_epoch, \
                                    train_loss=self.loss_tracker.get_avg_loss(), \
                                    duration=duration))
                if steps % steps_per_epoch == 0: 
                    self.current_epoch += 1
                    steps = 0
                    print ('VALIDATING ...')
                    self.validate()
                    self.model = self.model.train().cuda() 
                    self.classifier = self.classifier.train().cuda()
                    if self.stopping >= early_stopping: 
                        # Make sure to set number of epochs to max epochs
                        self.current_epoch = max_epochs
                    if self.check_end_train():
                        # Break the for loop
                        break
            if self.check_end_train(): 
                # Break the while loop
                break 
        print ('TRAINING : END') 
        print ('Training took {}\n'.format(datetime.datetime.now() - start_time))

class KneeTrainer(Trainer): 
    def __init__(self, model, architecture, optimizer, criterion, loss_tracker, save_checkpoint, classifier, save_best):
        super(KneeTrainer, self).__init__(model, architecture, optimizer, criterion, loss_tracker, save_checkpoint, classifier, save_best)

    def coalesce(self, y_pred, y_true, y_levels): 
        by_level = {'y_true': {}, 'y_pred': {}}
        for lvl in np.unique(y_levels): 
            by_level['y_pred'][lvl] = [] 
            by_level['y_true'][lvl] = [] 
        for i, pred in enumerate(y_pred): 
            by_level['y_pred'][y_levels[i]].append(pred)
            by_level['y_true'][y_levels[i]].append(y_true[i])
        for lvl in by_level['y_pred'].keys():
            by_level['y_pred'][lvl] = np.mean(by_level['y_pred'][lvl], axis=0)
            by_level['y_true'][lvl] = np.max(by_level['y_true'][lvl])
        y_pred_list = []
        y_true_list = []
        for lvl in by_level['y_pred'].keys():
            y_pred_list.append(by_level['y_pred'][lvl])
            y_true_list.append(by_level['y_true'][lvl])
        y_pred = np.vstack(y_pred_list)
        y_true = np.asarray(y_true_list)
        return y_pred, y_true

    def validate(self):
        with torch.no_grad():
            self.model = self.model.eval().cuda()
            self.classifier = self.classifier.eval().cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = [] ; y_levels = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                batch, labels, levels = data  
                output = self.model(batch.cuda())
                output = self.classifier(output.cuda())
                loss = self.criterion(output, labels.cuda())
                y_pred.append(torch.softmax(output, dim=1).cpu().numpy())
                y_true.extend(labels.numpy())
                y_levels.extend(levels.cpu().numpy())
                valid_loss += loss.item()
        y_pred = np.vstack(y_pred) 
        # Coalesce image-level predictions into patient-level
        y_pred, y_true = self.coalesce(y_pred, y_true, y_levels)
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, valid_loss)
        self.post_validate(valid_metric, metrics_dict)
