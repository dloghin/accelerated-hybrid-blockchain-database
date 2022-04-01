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

SRC_DIR="temp1"
DST_DIR="temp"
BENCH_LIST="ycsb_data"
WORKLOAD_LIST="workloada.dat workloadb.dat workloadc.dat"
BIN="../bin/preprocess"

mkdir -p $DST_DIR
for BENCH in $BENCH_LIST; do
    for WORKLOAD in $WORKLOAD_LIST; do
        mkdir -p $DST_DIR/$BENCH
        $BIN --load-read-path=$SRC_DIR/$BENCH/$WORKLOAD --load-write-path=$DST_DIR/$BENCH/$WORKLOAD --run-read-path=$SRC_DIR/$BENCH/run_$WORKLOAD --run-write-path=$DST_DIR/$BENCH/run_$WORKLOAD        
    done
done