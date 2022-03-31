#!/bin/bash

go build -o keccak-cpu keccak-cpu.go

#THREADS="2 4 6 8 10"
#for TH in $THREADS; do
#	./keccak-cpu --run-path ../scripts/temp/ycsb_data/run_workloada.dat --nthreads $TH
#done

./keccak-cpu --run-path ../scripts/temp/ycsb_data/run_workloada.dat --nthreads 6 --save

export LD_LIBRARY_PATH=../accelerator/sw/keccak256
make lib
go build -o keccak-acc keccak-acc.go
./keccak-acc --run-path ../scripts/temp/ycsb_data/run_workloada.dat --save

DIFF=`diff output-cpu.txt output-acc.txt`
if [ -z "$DIFF" ]; then
	echo "Test (CPU-only vs. FPGA) passed!"
else
	echo "Test (CPU-only vs. FPGA) failed!"
fi
