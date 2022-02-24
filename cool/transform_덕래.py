"""
필수적으로 해야 할 것!
mean과 std 평군 계산
추가적으로 적용 할 수 있는 사항
1. 하이퍼 파라미터(N,M,resize) 조정
2. 추가적인 augmentation 기술(cutmix, mixup 등등)
3. 찾아봅시다
"""

import augmentation.RandAugment
from torchvision import transforms

def get_train_transform(n = 3, magnitude = 10, resize = 384):
    return transforms.Compose([
        augmentation.RandAugment(n = n, m = magnitude),
        transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
    ])


def get_eval_transform(resize = 384):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
    ])