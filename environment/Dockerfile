FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu14.04

ARG user

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Update/install basic tools
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    graphviz \
    libhdf5-dev \
    sudo \
    vim \
    wget

RUN mkdir -p $CONDA_DIR && \
    cd $CONDA_DIR && \
    wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    chmod u+x Miniconda3-4.2.12-Linux-x86_64.sh && \
    ./Miniconda3-4.2.12-Linux-x86_64.sh -b -f -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV USER_UID 1000
RUN useradd -m -s /bin/bash -N -u $USER_UID $user && \
    echo "$user:$user" | chpasswd && adduser $user sudo && \
    chown -R $user $CONDA_DIR

USER $user

RUN mkdir $HOME/repos && \
    cd $HOME/repos && \
    git clone https://github.com/sallamander/dist-keras-tf.git && \
    chmod u+x dist-keras-tf/environment/install_conda_env.sh

ARG conda_version=4.2.13
RUN cd $HOME/repos && \
    ./dist-keras-tf/environment/install_conda_env.sh  $conda_version

RUN mkdir $HOME/keras
ADD keras.json /home/$user/keras/keras.json
