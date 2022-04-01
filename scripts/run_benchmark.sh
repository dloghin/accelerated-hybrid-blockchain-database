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

set -x

TSTAMP=`date +%F-%H-%M-%S`
LOGS="logs-hbdb-$TSTAMP"
mkdir $LOGS

N=$(($DEFAULT_NODES + 1))
DRIVERS=$DEFAULT_DRIVERS
WORKLOAD_FILE="$DEFAULT_WORKLOAD_PATH/$DEFAULT_WORKLOAD".dat
WORKLOAD_RUN_FILE="$DEFAULT_WORKLOAD_PATH/run_$DEFAULT_WORKLOAD".dat

# Generate server addresses. Server port is 1990
ADDRS="$IPPREFIX.2:1990"
for IDX in `seq 3 $N`; do
	ADDRS="$ADDRS,$IPPREFIX.$IDX:1990"
done

if [ $# -lt 1 ] || [ $1 != "all" ]; then
	DRIVERS=2
	THREADS="64"
fi

for TH in $THREADS; do
    ./restart_cluster.sh
    ./start_hbdb.sh   
    ../bin/hbdb-batch-bench --load-path=$WORKLOAD_FILE --run-path=$WORKLOAD_RUN_FILE --ndrivers=$DRIVERS --nthreads=$TH --server-addrs=127.0.0.1:1990 --key-file-prefix=client 2>&1 | tee $LOGS/hbdb-clients-$TH.txt
    SLOGS="$LOGS/hbdb-clients-$TH-logs"
    mkdir -p $SLOGS
    mv hbdb-server-1.log $SLOGS/    
    scp -o StrictHostKeyChecking=no root@$IPPREFIX.$(($N+1)):/kafka_2.12-2.7.0/zookeeper.log $SLOGS/
    scp -o StrictHostKeyChecking=no root@$IPPREFIX.$(($N+1)):/kafka_2.12-2.7.0/kafka.log $SLOGS/
done
