#!/bin/bash

ALGO=secp256k1
if [ $# -gt 0 ]; then
	ALGO=$1
fi

go build -o verify-cpu verify-cpu-$ALGO.go
go build -o verify-gpu verify-gpu-$ALGO.go

# Save the CPU results
# ./verify-cpu --run-path ../../scripts/temp/ycsb_data/run_workloada.dat --key-file client --nthreads 1 --save

THREADS="1 4 8 10 12 16"
for TH in $THREADS; do
	./verify-cpu --run-path ../../scripts/temp/ycsb_data/workloada.dat --key-file client --nthreads $TH
done
