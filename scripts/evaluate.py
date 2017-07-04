#!/usr/bin/env python

import os
import argparse
import yaml

from keras.models import Model

from data_iterators import DataSetIterator


def load_keras_model(model_dir):
    """Load the Keras model specified by the config and weights

    :param model_dir: str holding the model directory
    """

    fpath_config = os.path.join(model_dir, 'model.yml')
    fpath_weights = os.path.join(model_dir, 'weights.h5')

    with open(fpath_config, 'r') as f:
        model_config = yaml.load(f)

    model = Model.from_config(model_config['config'])
    model.load_weights(fpath_weights)

    return model


def validate_model_dir(model_dir):
    """Validate that the model_dir has the necessary files

    The model_dir should have a model.yml, config.yml, and weights.h5 file.

    :param model_dir: str holding the model directory
    """

    expected_files = ['model.yml', 'config.yml', 'weights.h5']

    for expected_file in expected_files:
        fpath_file = os.path.join(model_dir, expected_file)

        if not os.path.exists(fpath_file):
            msg = ('model_dir is expected to have a {} file but is missing '
                   'one!')
            raise FileNotFoundError(msg)


def validate_config(config):
    """Validate that the config specifies a dataset

    :param config: dct containing specs to use for training
    """

    dataset = config.get('dataset', {})
    name = dataset.get('name', None)
    if not name:
        msg = ('Must specify a dataset name via the \'dataset\' section in '
               'the config.')
        raise KeyError(msg)


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir', type=str,
        help=('Filepath to a model directory containing a model.yml, '
              'config.yml, and weights.h5 file.')
    )

    args = parser.parse_args()
    return args


def main():
    """Evaulate a Keras model"""

    args = parse_args()
    model_dir = args.model_dir

    model = load_keras_model(model_dir)

    fpath_config = os.path.join(model_dir, 'config.yml')
    with open(fpath_config, 'r') as f:
        config = yaml.load(f)
    dataset_name = config['dataset']['name']
    dataset_iterator = DataSetIterator(name=dataset_name)

    test_iter = dataset_iterator.get_test_iter()
    # evaluate on an aribtrary # of steps (10)
    model.compile(**config['compile_args'])
    result = model.evaluate_generator(test_iter, steps=10)
    print(result)


if __name__ == '__main__':
    main()
