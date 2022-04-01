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

. ./env.sh

N=$(($DEFAULT_NODES + 1))
CDIR=`pwd`

if [ $# -gt 0 ]; then
	N=$1
else
	echo -e "Usage: $0 <# containers>"
	echo -e "\tDefault: $N containers"
fi

IMGNAME="hbdb:latest"
PREFIX="hbdb"

DFILE=dockers.txt
rm -rf $DFILE

for idx in `seq 1 $N`; do
	CPUID=$(($idx+0))
	docker run -d --publish-all=true --cap-add=SYS_ADMIN --cap-add=NET_ADMIN --security-opt seccomp:unconfined --cpuset-cpus=$CPUID --name=$PREFIX$idx $IMGNAME tail -f /dev/null 2>&1 >> $DFILE	
done
while read ID; do
	docker exec $ID service ssh start
done < $DFILE
