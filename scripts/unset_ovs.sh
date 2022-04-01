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

if [ $EUID -ne 0 ]; then
	echo "This script must be run as root!"
	exit 1
fi

. ./env.sh

N=$(($DEFAULT_NODES + 1))
if [ $# -gt 0 ]; then
	N=$1
else
	echo -e "Usage: $0 <# containers>"
	echo -e "\tDefault: $N containers"
fi

PREFIX="hbdb"

for idx in `seq 1 $N`; do
	idx2=$(($idx+1))
	ovs-docker del-port ovs-br1 eth1 $PREFIX$idx
done
ovs-vsctl del-br ovs-br1
