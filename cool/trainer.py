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
from efficientnet_pytorch import EfficientNet


from cool.transform import get_transform
from cool.seed import seed_everything
from cool.Dataset import MaskDataset, ValDataset
from cool.split_by_kfold import Kfold
#from cool.utils import get_weighted_sampler
import cool.loss as ensemble_loss

class Trainer:
    
  def __init__(self, train_csv_path, train_img_path, save_param_path, seed):
      
    self.train_csv_path = train_csv_path
    self.train_img_path = train_img_path
    self.save_param_path = save_param_path
      
    os.makedirs(save_param_path, exist_ok=True)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.seed = seed
  
  
  # train_base는 train에 들어갈 것을 미리 구현해놓은 상태
  # target에 따라 예측하는 것이 변경될 수 있도록 설정
  
  def train_base(self, loader, model, optimizer, criterion, fold, epochs, max_limit, sub_dir, file_name):
    
    save_dir = os.path.join(self.save_param_path, sub_dir)
    # save_param/sub_dir/file_name.pt
    # 여기서 sub_dir은 각 target별 폴더가 달라지게 된다. ex) train_base_age / train_base_gender
    # file_name은 fold{몇번째 폴드인지 숫자}.pt
    # 그렇다면 각 fold마다의 validation 성적을 구한다.
    # epoch이 지속됨에 따라, 이전 epoch과 비교해 성적이 높아졌으면 저장을 하게 된다.
    
    os.makedirs(save_dir, exist_ok=True)
        
    best_score = 0
    best_epoch = 0
    patience_cnt = 0
        
    for sub_epoch in range(1, epochs+1):
      print(f'Epoch {sub_epoch}/{epochs}의 train을 시작합니다\n')
      
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
        
          print('{} mode에서 sub_epoch {} 에서의 acc : {:.4f}, loss : {:.3f}, f1-score : {:.4f}'.format(mode, sub_epoch, sub_epoch_acc, sub_epoch_loss, sub_f1))

        if mode == 'valid':
          if sub_f1 > best_score:
            best_score = sub_f1
            best_epoch = sub_epoch
            limit = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'{file_name}.pt'))
          
          elif limit == max_limit:
            print('\n훈련종료\n')
            print('validation set best f1 score는 epoch {}에서의 {:.4f}입니다.'.format(best_epoch, best_score))
            
            return

          limit += 1


    print('\n훈련종료\n')
    print('validation set best f1 score는 epoch {}에서의 {:.4f}입니다.\n'.format(best_epoch, best_score))
  
  
  
  
  
  
  def train(self, target, config, scheduler, pseudo_data=None):

        print(f'\n{target} train을 시작합니다.')

        folds = fold(train_csv_path = self.train_csv_path, train_img_path = self.train_img_path, random_state=self.seed , **config['fold'])

        for fold, (df_train, df_valid) in enumerate(customfold.folds, start = 1):
            seed_everything(self.seed)
            
            if pseudo_data is not None:
                df_train = pd.concat([pseudo_data, df_train])

            
            resize_input = EfficientNet.get_image_size(config['model'])
            # 우리의 모델에 들어가는 input size는 기존 pre-trained 모델의 input size을 기반으로 한다.
            
            transform_train = get_transform(augment=True,resize=resize_input, **config['transform'])
            transform_valid = get_transform(augment=False, resize=resize_input, **config['transform'])
            # 그것에 맞추어 사이즈 교체
            
            train_dataset = MaskDataset(df=df_train, transform=transform_train, target=target)
            valid_dataset = ValDataset(df=df_valid, transform=transform_valid, target=target)

            
            #sampler = get_weighted_sampler(label=df_train[target], ratio=config['sampler_size'])

            ## 여기 수정해야 한다.
            
            dataloaders = {'train' : DataLoader(train_dataset, sampler=sampler, **config['dataloader']),
                           'valid' : DataLoader(valid_dataset, drop_last=False, shuffle=False, **config['dataloader'])}
            
  
            #model = get_model(target=target)
            model.to(self.device)
            
            
            
            # criterion
            # loss 부분도 수정해주어야 한다.
            config_criterion = config['criterion']
            criterion = getattr(ensemble_loss, config_criterion['name'])(**config_criterion['parameters']) 
            

            # optimizer
            config_criterion = config['optimizer']
            optimizer = getattr(optim, config_criterion['name'])(model.parameters(), **config_criterion['parameters'])



            # train
            self.train_base(model= model, loader = dataloaders, criterion= criterion, optimizer= optimizer, 
                            sub_dir=f"{config['prefix_for_weight']}{target}", file_name=f'fold{fold}', **config['train'])

        return model, [os.path.join(self.weight_save_path, f'{config["prefix_for_weight"]}{target}', f'fold{fold}.pt') for fold in range(1,len(folds)+1)]
  
