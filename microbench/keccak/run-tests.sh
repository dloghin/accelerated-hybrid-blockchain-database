#!/bin/bash
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
