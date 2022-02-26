#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class Kfold:
    
    def __init__(self,n_splits,dir='/opt/ml/input/data/train/images'):
        self.dir=dir
        self.folds=[]
        self.n_splits=n_splits
        self.kfold()
        
    def kfold(self):
        skf = StratifiedKFold(n_splits=self.n_splits,shuffle=True)
        df=self.get_folder()
        for train_idx, test_idx in skf.split(df,df['label']): 
            self.folds.append((train_idx,test_idx))
            
      
    def get_folder(self):
        all=[]
        folders=os.listdir(self.dir)
        folders.sort()
        for folder in folders:
            if folder.startswith('.'):
                continue
            id, gender, _, age =folder.split('_')
            all.append([folder,self.labeling(gender,age)])
        df=pd.DataFrame(np.array(all),columns=['path','label'])
        return df

            
    def labeling(self,gender,age):
        gender_label =0
        age_label=0
        age=int(age)
        if gender == 'female':
            gender_label += 3
        if 27 <= age < 57:
            age_label += 1
        elif age >= 58:
            age_label += 2
        total_label=gender_label*100 + age_label
        
        return total_label  
            
        
    def __getitem__(self,idx):
        return self.folds[idx]
    def __len__(self):
        return len(folds)