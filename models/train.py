import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import tqdm
from torchvision.models import resnet34 as resnet
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from model import MultiTaskModel
from fastai import *
from fastai.vision import *
from torchvision import transforms
from torch import nn
from dataset import MaskBaseDataset, AugTrainTransform, TrainTransform
from loss import MultiTaskLossWrapper
import PIL 
#from DataBunch import DataBunch

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    #save_dir = increment_path(os.path.join(model_dir, args.name))
    save_dir = os.path.join(model_dir, args.name)
    
    target_list = ["mask", "gender", "age"]

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ##############################################################################################
    # -- dataset
    print('Start Augmentation.... 제발...\n')
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
            data_dir=data_dir
            )
    #aug_dataset = dataset_module(
    #        data_dir=data_dir
    #        )


    # target을 넘겨줘서 label과 num_classes가 변화하도록 구현해야 한다.
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        n = 2,
        magnitude = 9
    )

    


    #dataset.set_transform(transform)
    #aug_dataset.set_transform(AugTrainTransform(resize=args.resize))

    ##############################################################################################

    
    # # -- dataset
    # dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    # dataset = dataset_module(
    #         data_dir=data_dir
    #         )
    # # target을 넘겨줘서 label과 num_classes가 변화하도록 구현해야 한다.

    # # -- augmentation
    # transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     n = 2,
    #     magnitude = 9
    # )
    dataset.set_transform(transform)
    # print(len(dataset))

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    # tfms = get_transforms()
    # train_set.set_transform(tfms[0])
    # val_set.set_transform(tfms[1])
    #aug_train_set, aug_val_set = aug_dataset.split_dataset()
    #aug_train_set.indices = train_set.indices
    #aug_val_set.indices = val_set.indices
    


    
    #train_set = ConcatDataset([train_set, aug_train_set])
    #val_set = ConcatDataset([val_set, aug_val_set])
    
    

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    # model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model_module(
    #     num_classes=num_classes
    # ).to(device)
    def rmse_age(preds, age, gender, mask): return accuracy(preds[0],age)
    def acc_gender(preds, age, gender, mask): return accuracy(preds[1], gender)
    def acc_mask(preds, age, gender, mask): return accuracy(preds[2], mask)
    metrics = [rmse_age, acc_gender, acc_mask]


    model = MultiTaskModel(resnet, ps=0.25)
    #model = torch.nn.DataParallel(model)
    
    data = DataBunch(train_loader, val_loader)

    loss_func = MultiTaskLossWrapper(3).to(device)
    # -- loss & metric
    #criterion = create_criterion(args.criterion)  # default: cross_entropy

    learn = Learner(data, model, loss_func=loss_func, metrics=metrics)
    print([learn.model.encoder[:6],
             learn.model.encoder[6:],
             nn.ModuleList([learn.model.fc1, learn.model.fc2, learn.model.fc3])])
    #spliting the model so that I can use discriminative learning rates
    learn.split([learn.model.encoder[:6],
                    learn.model.encoder[6:],
                    nn.ModuleList([learn.model.fc1, learn.model.fc2, learn.model.fc3])])
        

    # if target == 'gender':
    #     scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # else:
    #     scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    # -- logging
    #logger = SummaryWriter(log_dir=os.path.join(save_dir, target))
    #with open(os.path.join(os.path.join(save_dir, target), 'config.json'), 'w', encoding='utf-8') as f:
        #json.dump(vars(args), f, ensure_ascii=False, indent=4)

#=====================================================================
# Training 시작
    best_val_acc = 0
    best_val_loss = np.inf

    learn.freeze()
    learn.lr_find()
    learn.fit_one_cycle(1,max_lr=1e-2,
                    callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='stage-1')])
    learn.load("stage-1")
    learn.unfreeze()
    learn.lr_find()
    learn.unfreeze()
    learn.fit_one_cycle(0,max_lr=slice(1e-6,3e-4),
                   callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='stage-2')])
    learn = learn.load("stage-2")

    trained_model = learn.model.cpu()
    torch.save(trained_model.state_dict(),"model_params_resnet34")


    class AgenethPredictor():
        def __init__(self, model):
            self.model = model
            self.tfms = get_transforms()[1]
            self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #imagenet stats
            self.age = {0:0, 1:1, 2:2}
            self.gender = {0:0,1:3}
            self.mask = {0:0,1:6,2:12}
            #self.trans = transforms.ToPILImage()
        def predict(self,x):
            #x is a PIL Image
            x = Image(pil2tensor(x, dtype=np.float32).div_(255))
            x = x.apply_tfms(self.tfms, size = 64)
            x = self.norm(x.data)

            preds = self.model(x.unsqueeze(0))
            
            age = self.age[torch.softmax(preds[0],1).argmax().item()]
            gender = self.gender[torch.softmax(preds[1],1).argmax().item()]
            mask = self.mask[torch.softmax(preds[2],1).argmax().item()]
            return age, gender, mask

    ageneth_predictor = AgenethPredictor(trained_model)


    data_dir = "/opt/ml/input/data/eval"
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    preds = []
    cnt=0
    for img_path in img_paths :
        img = PIL.Image.open(img_path)
        age,gender,mask = ageneth_predictor.predict(img)
        total=age+gender+mask
        preds.append(total)
        cnt+=1
    info['ans']=preds
    info.to_csv(os.path.join(output_dir, f'output.csv'),index=False)
    print(f'Inference Done!')        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#    from dotenv import load_dotenv
    import os
#    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='TrainTransform', help='data augmentation type (default: TrainTransform)')
    #parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument("--resize", nargs="+", type=list, default=(224, 224), help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    
    parser.add_argument('--model', type=str, default='coatnet', help='model type (default: BaseModel)')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='MultiTaskLossWrapper', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=3, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)


    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)

