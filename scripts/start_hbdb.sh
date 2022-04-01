#!/bin/bash

. ./env.sh

set -x

echo "HBDB Accelerated supports only 1 node!"
# 1 node + 1 Kafka node
N=2
BLKSIZE=100

# Kafka
KAFKA_ADDR=$IPPREFIX".$(($N+1))"
ssh -o StrictHostKeyChecking=no root@$KAFKA_ADDR "cd /kafka_2.12-2.7.0/config && echo 'advertised.listeners=PLAINTEXT://$KAFKA_ADDR:9092' >> server.properties"
ssh -o StrictHostKeyChecking=no root@$KAFKA_ADDR "cd /kafka_2.12-2.7.0; bin/zookeeper-server-start.sh config/zookeeper.properties > zookeeper.log 2>&1 &"
sleep 10s
ssh -o StrictHostKeyChecking=no root@$KAFKA_ADDR "cd /kafka_2.12-2.7.0; bin/kafka-server-start.sh config/server.properties > kafka.log 2>&1 &"
sleep 10s
ssh -o StrictHostKeyChecking=no root@$KAFKA_ADDR "cd /kafka_2.12-2.7.0; bin/kafka-topics.sh --create --topic shared-log --bootstrap-server 0.0.0.0:9092"
sleep 10s

# Nodes
NODES=node1
for I in `seq 1 $(($N-1))`; do
        NODES="$NODES,node$I"
done

RESPATH="../accelerator/bin"
BINPATH="../bin"

# Start
for I in `seq 1 $(($N-1))`; do
	ADDR=$IPPREFIX".$(($I+1))"
	ssh -o StrictHostKeyChecking=no root@$ADDR "cd /; redis-server --protected-mode no > redis.log 2>&1 &"
        LD_LIBRARY_PATH=$RESPATH $BINPATH/hbdb-server --signature=node$I --parties=${NODES} --blk-size=$BLKSIZE --addr=:1990 --kafka-addr=$KAFKA_ADDR:9092 --kafka-group=$I --kafka-topic=shared-log --redis-addr=$ADDR:6379 --redis-db=0 --ledger-path=ledger$I --xclbin-path=$RESPATH/keccak256_kernel.xclbin > hbdb-server-$I.log 2>&1 &
done
