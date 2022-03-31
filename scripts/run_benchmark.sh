#!/bin/bash

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
	THREADS="4"
fi

for TH in $THREADS; do
    ./restart_cluster.sh
    ./start_hbdb.sh
    # Run with simple (sigle) requests
    #../bin/hbdb-bench --load-path=$WORKLOAD_FILE --run-path=$WORKLOAD_RUN_FILE --ndrivers=$DRIVERS --nthreads=$TH --server-addrs=$ADDRS 2>&1 | tee $LOGS/hbdb-clients-$TH.txt
    # Run with batch requests
    ../bin/hbdb-batch-bench --load-path=$WORKLOAD_FILE --run-path=$WORKLOAD_RUN_FILE --ndrivers=$DRIVERS --nthreads=$TH --server-addrs=$ADDRS 2>&1 | tee $LOGS/hbdb-clients-$TH.txt
    # copy logs
    SLOGS="$LOGS/hbdb-clients-$TH-logs"
    mkdir -p $SLOGS
    for I in `seq 2 $N`; do
	    IDX=$(($I-1))
	    scp -o StrictHostKeyChecking=no root@$IPPREFIX.$I:/hbdb-server-$IDX.log $SLOGS/
    done
    scp -o StrictHostKeyChecking=no root@$IPPREFIX.$(($N+1)):/kafka_2.12-2.7.0/zookeeper.log $SLOGS/
    scp -o StrictHostKeyChecking=no root@$IPPREFIX.$(($N+1)):/kafka_2.12-2.7.0/kafka.log $SLOGS/
done
