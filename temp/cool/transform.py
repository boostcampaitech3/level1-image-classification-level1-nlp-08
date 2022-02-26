from cool import augmentation
from torchvision import transforms

class train_transform():
    def __init__(self, n, magnitude, resize):
        self.transform = transforms.Compose([
            augmentation.RandAugment(n = n, m = magnitude),
            transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        return self.transform(img)
    
    
class eval_transform():
    def __init__(self, resize):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        return self.transform(img)