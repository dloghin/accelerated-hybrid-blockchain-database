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

set -x

# List of number of nodes
NODES="4 8 16 32 64"

# Default number of nodes
DEFAULT_NODES=1

# IP prefix of the containers
IPPREFIX="192.168.30"

# List of number of threads (clients)
THREADS="4 8 16 32 64 128 192 256"

# List of workloads
WORKLOADS="workloada workloadb workloadc"

# Default workload
DEFAULT_WORKLOAD="workloada"

# Default workload path
DEFAULT_WORKLOAD_PATH="temp/ycsb_data"

# List of YCSB distributions
DISTRIBUTIONS="uniform latest zipfian"

# Transaction delay times (in ms)
TXDELAYS="0 10 100 1000"

# Transaction (record) sizes
TXSIZES="512B 2kB 8kB 32kB 128kB"

# Block sizes
BLKSIZES="10 100 1000 10000"

# Default block size
DEFAULT_BLOCK_SIZE="1000"

# Benchmark
DEFAULT_DRIVERS=2
DEFAULT_THREADS=256
