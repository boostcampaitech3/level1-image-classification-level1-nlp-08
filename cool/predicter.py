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


from cool.transform import eval_transform
from cool import models
from cool.dataset import TestDataset
from cool import loss


class Predicter:
    def __init__(self, eval_csv_path, eval_img_path):
      
        self.eval_csv_path = eval_csv_path
        self.eval_img_path = eval_img_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.df_eval = pd.read_csv(self.eval_csv_path)

    def predict(self, config):
        dataloader = self.get_dataloader(config)
        weights = config['weights']

        with torch.no_grad():
            for target in weights:
                
                #geattr에 importmodule 활용시 에러 발생한다고 함
                model_module = getattr(models, config['model'])
                model = model_module(num_classes=2 if target=='gender' else 3)
                model.to(self.device)
                model.eval()
                
                for weight_path in weights[target]:
                    prob_batch = []
                    model.load_state_dict(torch.load(weight_path))

                    for inputs in dataloader:
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        
                        self.df_eval[target] = self.ouputs.argmax(axis = -1)

        df_submit = self.calc_target(self.df_eval)
        df_submit.to_csv('output.csv', index=False)

                # probs = []
                # for weight_path in weights[target]:

                #     prob = []
                #     model.load_state_dict(torch.load(weight_path))

                #     for inputs in tqdm(dataloader):
                #         inputs = inputs.to(self.device)
                #         outputs = model(inputs)
                #         prob_batch =F.softmax(outputs, dim=-1)
                #         prob.append(prob_batch.cpu().numpy())

                #     probs.append(np.concatenate(prob, axis=0))
                
                # ensembled_prob = np.mean(probs, axis=0)
                # soft_vote = ensembled_prob.argmax(axis=-1)
                # df_eval[target] = soft_vote

        # df_submit = self._sum_targets(df_eval)
        # df_submit.to_csv('output.csv', index=False)
  
        
    def calc_target(self, df):
        df['ans'] = df['age'] + 3*df['gender'] + 6*df['mask']
        return df[['ImageID', 'ans']]
    
    def get_dataloader(self, config):
        self.df_eval['path'] = self.df_eval['ImageID'].apply(lambda x: os.path.join(self.eval_img_path, x))
        
        # transform
        resize = 224
        transform_eval = eval_transform(resize = resize)

        # dataset
        eval_dataset = TestDataset(img_dir = "/opt/ml/input/data/eval/images", transform=transform_eval)

        # dataloader
        dataloader = DataLoader(eval_dataset, drop_last=False, shuffle=False, **config['dataloader'])
        
        return dataloader

    # def get_pseudo_label(self, target, weight_paths, config, model=None):
        
    #     df_eval = pd.read_csv(self.eval_csv_path)
    #     dataloader = self._get_dataloader(config)        
        
    #     if model is None:
    #         model = EfficientNet.from_pretrained(config['model'], num_classes=2 if target=='gender' else 3)
    #         model.to(self.device)
    #     model.eval()

    #     probs = []

    #     with torch.no_grad():
    #         for weight_path in weight_paths:

    #             prob = []
    #             model.load_state_dict(torch.load(weight_path))

    #             for inputs in tqdm(dataloader):
    #                 inputs = inputs.to(self.device)
    #                 outputs = model(inputs)
    #                 prob_batch =F.softmax(outputs, dim=-1)
    #                 prob.append(prob_batch.cpu().numpy())

    #             probs.append(np.concatenate(prob, axis=0))
            
    #     ensembled_prob = np.mean(probs, axis=0)
    #     soft_vote = ensembled_prob.argmax(axis=-1)
    #     idx_filtered = np.nonzero(np.any(ensembled_prob >= np.array(config['threshold']), axis=1))[0]

    #     df_eval[target] = soft_vote
    #     df_filtered = df_eval.iloc[idx_filtered].copy()
    #     print('pseudo labeled datas:', len(df_filtered))

    #     return df_filtered
