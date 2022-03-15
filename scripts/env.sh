#!/bin/bash

set -x

# List of number of nodes
NODES="4 8 16 32 64"

# Default number of nodes
DEFAULT_NODES=4

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
DEFAULT_DRIVERS=4
DEFAULT_THREADS=256
