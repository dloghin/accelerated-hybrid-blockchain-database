#!/bin/bash

go build -o verify-cpu verify-cpu.go

# Save the CPU results
# ./verify-cpu --run-path ../../scripts/temp/ycsb_data/run_workloada.dat --key-file client --nthreads 1 --save

THREADS="2 4 6 8 10"
for TH in $THREADS; do
	./verify-cpu --run-path ../../scripts/temp/ycsb_data/run_workloada.dat --key-file client --nthreads $TH
done