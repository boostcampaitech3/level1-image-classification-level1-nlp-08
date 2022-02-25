from cool.trainer import Trainer
from cool.predicter import Predicter

import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_pred", type=str, help ='path of predicter configuration yaml')
    args = parser.parse_args()

    with open(args.config_pred) as f:
        config_pred = yaml.load(f, Loader = yaml.FullLoader)

    predicter = predicter(**config_pred['predicter'])
    Predicter.predict(config_pred['predict'])

    print('Evaluation Complete!')