#!/bin/bash

CONDA_VERSION=$1

conda install conda=$CONDA_VERSION

REPO_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
conda env create -f $REPO_ENV_DIR/environment.yml
source activate dist-keras
pip install -e $REPO_ENV_DIR/../dist-keras-tf/

TF_WHEEL_URL="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl"
TF_WHEEL_DIR=/tmp/tf_pip_wheel
TF_WHEEL_PATH=$TF_WHEEL_DIR/$(basename $TF_WHEEL_URL)

mkdir $TF_WHEEL_DIR
cd $TF_WHEEL_DIR
wget $TF_WHEEL_URL
pip install $TF_WHEEL_PATH

cd $HOME
