from .model_utils import PACKAGE_NAME, ModelConfig
import yaml
from argparse import ArgumentParser

parser = ArgumentParser(prog=f'{PACKAGE_NAME}.create_config_file',
    description='Creating a model configuration file for model serving.')

parser.add_argument('-n', '--num-models', type=int, default=1, help='Number of model examples in the config file.')

def main():
    args = parser.parse_args()
    num_models = args.num_models
    configs = [ModelConfig(model_id_or_path=f'/path/to/model_{i}')._asdict() for i in range(num_models)]
    with open('model_configs.yaml', 'w') as f:
        f.write(yaml.dump(configs, sort_keys=False))

if __name__ == '__main__':
    main()
