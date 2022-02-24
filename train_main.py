import argparse
import yaml
from cool.trainer import Trainer
from cool.predicter import Predicter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Spam classifier')
    
    parser.add_argument('--config_train', type=str, 
                        help='path to train yaml')
    parser.add_argument('--config_infer', type=str, 
                        help='path oto inference yaml') 
    
    args = parser.parse_args()

    with open(args.config_train, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(args.config_infer, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    trainer = Trainer(**config['trainer'])
    predicter = Predicter(**config['predicter'])
    
    
    for target in config_train['targets']:
        base_model, weight_paths = trainer.train(target, config_train[f'train_{target}'])
        df_pseudo = predicter.get_pseudo_label(target, weight_paths, config_infer[f'pseudo_labeling_{target}'], model=base_model)
        model, weight_paths = trainer.train(target, config_train[f'train_with_pseudo_{target}'], df_pseudo=df_pseudo)
    
    print('Model train Complete')