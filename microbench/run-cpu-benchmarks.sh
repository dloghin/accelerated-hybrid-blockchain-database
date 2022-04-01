#!/bin/bash

go build -o keccak-secp256k1-cpu keccak-secp256k1-cpu.go

THREADS="2 4 6 8 10"
for TH in $THREADS; do
	./keccak-secp256k1-cpu --run-path ../scripts/temp/ycsb_data/run_workloada.dat --nthreads $TH --key-file-prefix client
done
