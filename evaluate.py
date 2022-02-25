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

    predicter = Predicter(**config_pred['predicter'])
    predicter.predict(config_pred['predict'])

    print('예측 및 submission csv 생성 완료')