#!/usr/bin/env python

import argparse
import os
import yaml

from data_iterators import DataSetIterator
from networks import VGG16
from trainer import KerasTrainer


def save_config(config, output_dir):
    """Save the config into the output_dir

    :param config: dct containing specs to use for training
    :param output_dir: str holding the output directory to save the
    ig to
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fpath_config = os.path.join(output_dir, 'config.yml')
    with open(fpath_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def validate_config(config):
    """Validate the config has the necessary specs to train

    The config needs to specify a loss via a \'compile_args\' section as well
    as a dataset name via a \'dataset\' section. Raise a value error if either
    of these are not present.

    :param config: dct containing specs to use for training
    """

    compile_args = config.get('compile_args', {})
    loss = compile_args.get('loss', None)
    if not loss:
        msg = ('Must specify a loss via the \'compile_args\' section in order '
               'to train!')
        raise ValueError(msg)

    dataset = config.get('dataset', {})
    name = dataset.get('name', None)
    if not name:
        msg = ('Must specify a dataset name via the \'dataset\' section in order '
               'to train!')
        raise ValueError(msg)


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', type=str,
        help=('Filepath to a model training config (YML file). Must specify '
              'a loss via a \'compile_args\' section and a dataset name via a '
              '\'dataset\' section.')
    )
    parser.add_argument(
        '--gpu_ids', type=int, nargs='+',
        help='GPU ID for the GPU to train the model on.'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Filepath to the directory to store model output in.'
    )

    args = parser.parse_args()
    return args


def main():
    """Main function to train a model"""

    args = parse_args()
    fpath_config = args.config
    output_dir = args.output_dir

    with open(fpath_config, 'r') as f:
        config = yaml.load(f)
    validate_config(config)
    save_config(config, output_dir)

    dataset_name = config['dataset']['name']
    dataset_iterator = DataSetIterator(name=dataset_name)

    # we know that the validate_config function ensures that there is at least
    # a loss specified in compile_args, and the KerasTrainer class specifies
    # defaults for fit_args if None
    compile_args = config['compile_args']
    fit_args = config.get('fit_args', None)

    # NOTE: This is the only network supported right now, but in the future
    # it might be configurable via the config
    network = VGG16()
    trainer = KerasTrainer(output_dir=output_dir)
    trainer.train(
        network, dataset_iterator,
        compile_args, fit_args, 
        args.gpu_ids
    )


if __name__ == '__main__':
    main()
