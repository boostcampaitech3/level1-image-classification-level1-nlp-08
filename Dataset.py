
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image


import pandas as pd
import numpy as np
import os


def labelling(name):
  label = 0
  info, mask_type = name.split('/')[-2:]
  info = info.split('_')
  gender, age = info[1], int(info[3])
  if 'incorrect' in mask_type:
    label += 6
  elif 'normal' in mask_type:
    label += 12
  
  if gender == 'female':
    label += 3
  
  if 27 <= age < 57:
    label += 1
  elif age >= 58:
    label += 2
  
  return label


train_data_root = '/opt/ml/input/data/train/train.csv'
train_image_root= '/opt/ml/input/data/train/images'


train_data = pd.read_csv(train_data_root)

drop_features=['race']
train_data=train_data.drop(columns=drop_features)

df = pd.DataFrame(columns=['path','mask','gender','age'])

###########################
for i in range(2700):
    name=train_data['path'][i]
    titles=name.split('_')
    file_list = os.listdir(train_image_root+'/'+name)
    for file in file_list:
        if 'mask' in file or 'normal' in file:
            if '._' in file:
                continue
            if 'incorrect' in file:
                a=pd.DataFrame(data=[[name+'/'+file,'Incorrect',titles[1],titles[3]]], columns=['path','mask','gender','age'])
            elif 'normal' in file:
                a=pd.DataFrame(data=[[name+'/'+file,'Not wear',titles[1],titles[3]]], columns=['path','mask','gender','age'])
            else:
                a=pd.DataFrame(data=[[name+'/'+file,'Wear',titles[1],titles[3]]], columns=['path','mask','gender','age'])

            df=df.append(a)
############################
        
display(df)



class MaskDataset(Dataset):
    def __init__(self,df,transform,target):
        self.df = df
        self.transform=transform
        self.target=target
        self.classes = df.columns
        
    def __getitem__(self,idx):
        image = Image.open(train_image_root+'/'+self.df['path'].iloc[idx])
        image = self.transform(image)
        label = self.df[self.target].iloc[idx]
        if self.target == 'age': 
            ###########################
            label=int(label)
            if 0<= label < 30 :
                label='<30'
            elif 30<= label < 60 :
                label='30<= and <60'
            else :
                label='60<'
            ###########################  
        
        return image, label
    
    def __len__(self):
        return len(self.df)

    
    
class EvalDataset(Dataset):
    def __init__(self,df,transform):
        self.df=df
        self.transform=transform
        
    def __get_item__(self,idx):
        image = Image.open(train_image_root+'/'+self.df['path'].iloc[idx])
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.df)
    



dataset = MaskDataset(df,transforms.ToTensor(),'mask')
len(dataset)




sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = range(len(dataset))

val = [y for _, y in dataset]



(train_index, val_index), = sss.split(indices, val)
    
train_dataset = Subset(dataset, train_index)
val_dataset = Subset(dataset, val_index)


len(train_dataset)



dataloader_train_mask = DataLoader(dataset=train_dataset,
                                    batch_size=5,
                                    shuffle=True,
                                    num_workers=1,
                                   )
dataloader_val_mask = DataLoader(dataset=val_dataset,
                                    batch_size=5,
                                    shuffle=True,
                                    num_workers=1,
                                   )


next(iter(dataloader_train_mask))


next(iter(dataloader_val_mask))





