#!/bin/bash
#
# Copyright 2022 Dumitrel Loghin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

MYUSER=`whoami`

# Docker - https://docs.docker.com/engine/install/ubuntu/
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo adduser $MYUSER docker

# OpenVSwitch, KafkaCat, jq, make, gcc, java, pip
sudo apt -y install net-tools openvswitch-switch kafkacat jq make gcc default-jdk python3-pip

# Go 1.16.10
mkdir -p temp
cd temp
wget https://golang.org/dl/go1.16.10.linux-amd64.tar.gz
tar xf go1.16.10.linux-amd64.tar.gz
mkdir gopath
GOROOT=`pwd`/go
GOPATH=`pwd`/gopath
echo "" >> /home/$MYUSER/.bashrc
echo "export GOROOT=$GOROOT" >> /home/$MYUSER/.bashrc
echo "export GOPATH=$GOPATH" >> /home/$MYUSER/.bashrc
echo "export PATH=$PATH:$GOROOT/bin" >> /home/$MYUSER/.bashrc

echo "*** Please log out or reboot your system!"
