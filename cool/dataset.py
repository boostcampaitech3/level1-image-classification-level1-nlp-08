from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image

import pandas as pd
import numpy as np
import os


#dir= '/opt/ml/input/data/train/images'
#dataset = MaskDataset(dir,transforms.ToTensor())

class MaskDataset(Dataset):
    def __init__(self, dir, transform, X=np.array(range(2700)), target=None):
        self.dir = dir
        self.target = target
        self.transform = transform
        self.X = X
        self.df = self.get_df()
        self.classes = self.df.columns
        
        
    def get_df(self):
        all_files = []
        folders = os.listdir(self.dir)
        folders.sort()
        folders.reverse()
        for i in self.X:
            if folders[i].startswith('.'):
                continue
            #id, gender, race, age =folders[i].split('_')
            id, gender, _, age =folders[i].split('_')

            img_dir = os.path.join(self.dir,folders[i])
            
            for imgname in os.listdir(img_dir):
                if imgname.startswith('.'):
                    continue
                path = os.path.join(img_dir,imgname)
                mask_label, gender_label, age_label, total_label = self.labeling(imgname, gender, age)
                all_files.append([path, mask_label, gender_label, age_label, total_label])
                
        df=pd.DataFrame(np.array(all_files), columns = ['path', 'mask', 'gender', 'age', 'label'])
        return df
    
    def labeling(self, imgname, gender, age):
        mask_label = 0
        gender_label = 0
        age_label = 0
        age = int(age)
        
        if 'incorrect' in imgname:
            mask_label += 1
        elif 'normal' in imgname:
            mask_label += 2
            
        if gender == 'female':
            gender_label += 1
        if 27 <= age < 57:
            age_label += 1
        elif age >= 58:
            age_label += 2

        total_label = mask_label*6 + gender_label*3 + age_label
        
        return mask_label, gender_label, age_label, total_label
    
    def __getitem__(self, idx):
        image = Image.open(self.df['path'].iloc[idx])
        image = self.transform(image)
        if self.target is not None:
            label = self.df[self.target].iloc[idx]
        else:          
            label = self.df['label'].iloc[idx]
        return image, label
    
    def __len__(self):
        return len(self.df)

#################################

class ValDataset(Dataset):
    def __init__(self, dir, transform, y=np.array(range(2700))):
        self.dir = dir
        self.transform = transform
        self.y = y
        self.df = self.get_df()
        self.classes = self.df.columns
        
        
    def get_df(self):
        all_files = []
        folders = os.listdir(self.dir)
        folders.sort()
        folders.reverse()
        for i in self.y:
            if folders[i].startswith('.'):
                continue

            img_dir = os.path.join(self.dir, folders[i])
            
            for imgname in os.listdir(img_dir):
                if imgname.startswith('.'):
                    continue
                path = os.path.join(img_dir, imgname)
                all_files.append(path)
                
        df = pd.DataFrame(np.array(all_files),columns = ['path'])
        return df
    
    
    def __getitem__(self,idx):
        image = Image.open(self.df['path'].iloc[idx])
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.df)
    

#################################

class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.images = os.listdir(os.path.join(self.img_dir))
        self.transform = transform
        
    def __get_item__(self,idx):
        image = Image.open(self.img_dir + '/' + self.images[idx])
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)