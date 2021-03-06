import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import random
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from cool.seed import seed_everything
from cool.dataset import MaskDataset
from cool.split_by_kfold import Kfold


from cool.transform import train_transform
from cool.transform import eval_transform
from cool import models
from cool import loss

class Trainer:
    
  def __init__(self, train_csv_path, train_img_path, weight_save_path, seed):
      
    self.train_csv_path = train_csv_path
    self.train_img_path = train_img_path
    self.weight_save_path = weight_save_path
    self.best_score = 0
    self.best_epoch = 0
    self.limit = 0
    
      
    os.makedirs(weight_save_path, exist_ok=True)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.seed = seed
  
  
  # train_base는 train에 들어갈 것을 미리 구현해놓은 상태
  # target에 따라 예측하는 것이 변경될 수 있도록 설정
  
  def train_base(self, loader, model, optimizer, criterion, sub_dir, file_name, sub_epoch, max_limit, fold):
    '''
    loader : dataloader
    model : configs.train.yaml -> train_base
    optimizer : config['optimizer']
    criterion : config['criterion']
    fold : StratifiedKFold
    max_limits : For Early Stopping
    sub_dir : Saved weight
    '''

    
    save_dir = os.path.join(self.weight_save_path, sub_dir)
    # save_param/sub_dir/file_name.pt
    # 여기서 sub_dir은 각 target별 폴더가 달라지게 된다. ex) train_base_age / train_base_gender
    # file_name은 fold{몇번째 폴드인지 숫자}.pt
    # 그렇다면 각 fold마다의 validation 성적을 구한다.
    # epoch이 지속됨에 따라, 이전 epoch과 비교해 성적이 높아졌으면 저장을 하게 된다.
    
    os.makedirs(save_dir, exist_ok=True)
        
      
    train_mode = ['train', 'valid']
      
    for mode in train_mode:
      if mode == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0
      running_cnt = 0
      y_true, y_pred = [], []
      
      for inputs, labels in tqdm(loader[mode]):
        inputs = inputs.to(self.device)
        labels = torch.LongTensor(list(map(int, labels)))
        labels = labels.to(self.device)
        
        with torch.set_grad_enabled(mode == 'train'): 
            # True, False가 들어가주어야 한다.
            # 자동으로 require_grad가 수정
            
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          
          y_true.extend(labels.tolist())
          y_pred.extend(preds.tolist())
          
          if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
          running_cnt += inputs.size(0)
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
        
        sub_epoch_loss = running_loss / running_cnt
        sub_epoch_acc = running_corrects.double() / running_cnt
        sub_f1 = f1_score(y_true, y_pred, average='macro')
        
      print('{} mode에서 sub_epoch {} 에서 {} fold에서 acc : {:.4f}, loss : {:.3f}, f1-score : {:.4f}'.format(mode, sub_epoch, fold, sub_epoch_acc, sub_epoch_loss, sub_f1))

      if mode == 'valid':
        if sub_f1 > self.best_score:
          self.best_score = sub_f1
          self.best_epoch = sub_epoch
          self.limit = 0
          torch.save(model.state_dict(), os.path.join(save_dir, f'{file_name}.pt'))
          
        elif self.limit == max_limit:
          print('\n훈련종료\n')
          print('validation set best f1 score는 epoch {}에서의 {:.4f}입니다.'.format(self.best_epoch, self.best_score))
            
          return

        self.limit += 1


    
  
  
  
  
  
  
  #def train(self, target, config, scheduler, pseudo_data=None):
  def train(self, target, config, pseudo_data=None):
        #target = 'label' or 'gender', 'age','mask'
        #kfold안에 들어갈 n_split 설정 필요
    print(f'\n{target} train을 시작합니다.')    
    epochs = config['train']['epochs']
    max_limit = config['train']['max_limit']
    
    for sub_epoch in range(1, epochs+1):
      
      print(f'Epoch {sub_epoch}/{epochs}의 train을 시작합니다\n')
      
      self.best_score = 0
      self.best_epoch = 0

      folds = Kfold(2).folds

      for fold, (train_idx, val_idx) in enumerate(folds, start = 1):
        seed_everything(self.seed)
            
        if pseudo_data is not None:
          df_train = pd.concat([pseudo_data, df_train])


        resize_input = 224 # train.yaml에 define
        # 우리의 모델에 들어가는 input size는 기존 pre-trained 모델의 input size을 기반으로 한다.
            
        transform_train = train_transform(**config['transform'])
        transform_valid = eval_transform(resize=resize_input)
        # 그것에 맞추어 사이즈 교체
            
        train_dataset = MaskDataset(dir=self.train_img_path, transform=transform_train, X=train_idx,target=target)
        valid_dataset = MaskDataset(dir=self.train_img_path, transform=transform_valid, X=val_idx, target=target)

            
        ## 여기 수정해야 한다.
            
        dataloaders = {'train' : DataLoader(train_dataset, **config['dataloader']),
                      'valid' : DataLoader(valid_dataset, drop_last=False, shuffle=False, **config['dataloader'])}
            
  
        model_module = getattr(models, config['model'])
        model = model_module(num_classes=2 if target=='gender' else 3)
        model.to(self.device)
            
            
            
        # criterion
        # loss 부분도 수정해주어야 한다.
        config_criterion = config['criterion']
        criterion = getattr(loss, config_criterion['name'])(**config_criterion['parameters'])
            

        # optimizer
        config_optimizer = config['optimizer']
        optimizer = getattr(optim, config_optimizer['name'])(model.parameters(), **config_optimizer['parameters'])



        # train
        self.train_base(loader = dataloaders, model= model, optimizer= optimizer, criterion= criterion,
                        sub_dir=f"{config['prefix_for_weight']}{target}", file_name=f'fold{fold}', sub_epoch=sub_epoch, max_limit=max_limit, fold = fold)
    
    print('\n훈련종료\n')
    print('validation set best f1 score는 epoch {}에서의 {:.4f}입니다.\n'.format(self.best_epoch, self.best_score))
        
    return model, [os.path.join(self.weight_save_path, f'{config["prefix_for_weight"]}{target}', f'fold{fold}.pt')]