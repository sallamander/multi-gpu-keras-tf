#!/bin/bash


# keep track of completed setup steps via an env directory
if [[ ! -d ~/env ]]; then
    mkdir ~/env
fi


# =====================
# === NVIDIA Driver ===
# =====================

install_nvidia_driver(){
    sudo apt-get update
    sudo apt-get upgrade -y

    echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
    echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf

    sudo update-initramfs -u
    sudo apt-get install -y linux-image-extra-virtual
    sudo apt-get install -y linux-source linux-headers-`uname -r`

    cd /tmp
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/361.93.02/NVIDIA-Linux-x86_64-361.93.02.run
    sudo bash ./NVIDIA-Linux-x86_64-361.93.02.run --ui=none --no-questions --accept-license
}

if [[ -f ~/env/1-NVIDIA-DONE ]]; then
    echo 'NVIDIA driver already installed'
else
    echo 'Installing NVIDIA driver, will reboot when complete...'
    install_nvidia_driver

    echo 'NVIDIA driver install complete.' > ~/env/1-NVIDIA-DONE
    sudo reboot now
fi

# ======================================
# === Docker & Nvidia-Docker Install ===
# ======================================

install_docker(){
    # all from Docker' wesbite
    sudo apt-get install -y \
      apt-transport-https \
      ca-certificates \
      curl \
      software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository \
      "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"

    sudo apt-get update
    sudo apt-get install -y docker-ce=17.03.1~ce-0~ubuntu-trusty
    sudo usermod -aG docker $USER
}

install_nvidia_docker(){
    # from nvidia-docker GH: https://github.com/NVIDIA/nvidia-docker
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
}

if [[ -f ~/env/2-DOCKER-INSTALL-DONE ]]; then
    echo 'Docker env already installed'
else
    echo 'Installing Docker env...'
    install_docker
    install_nvidia_docker

    echo 'Docker & Nvidia-Docker install complete.' > ~/env/2-DOCKER-INSTALL-DONE
    sudo reboot
fi

# ==============================
# === Docker Container Setup ===
# ==============================

if [[ -f ~/env/3-DOCKER-CONTAINER-DONE ]]; then
    echo 'Docker container all set up'
    exit 0
else
    echo 'Setting up Docker container...'
    cd $HOME/repos/dist-keras-tf/environment
    docker build --build-arg user=$USER -t keras-dist -f Dockerfile .
    echo 'Docker container setup complete.' > ~/env/3-CONTAINER-DONE
fi
