# Accelerated Hybrid Blockchain Database

This project is described on [hackster.io](https://www.hackster.io/dumiloghin/hybrid-blockchain-database-system-with-xilinx-varium-c1100-25441e).

## Overview

This project accelerates a Hybrid Blockchain Database (HBDB) by offloading the Keccak256 digest 
generation and ECDSA signature verification to an FPGA. The code was tested on a Xilinx Varium C1100.

![Approach Diagram](res/diagram.png?raw=true "Approach Diagram")

The preliminary results show that FPGA acceleration achieves excellent performance (green bar). 
It has almost the same performance as HBDB without signatures (blue bar, baseline 1), and 2X performance 
over HBDB with signatures running only on the CPU (red bar, baseline2).

![Results](res/results.png?raw=true "Results")

The code for baseline1 and baseline2 is on the homonym branches.

The code of this project started from [this code](https://github.com/nusdbsystem/Hybrid-Blockchain-Database-Systems). Many thanks to the colleagues that worked on the original version of the code.

## Build and Run

### Pre-requisites

- Install dependencies such as Docker, go, Java (for YCSB)
- Download and prepare YCSB data
- Build binaries
- Build docker images

```
# in the root of the repo
go mod tidy
cd scripts
./install_dependencies.sh
# build libkeccak256 (see below)
./build_binaries.sh
./gen_ycsb_data.sh 
mv temp temp1
./preprocess_ycsb_data.sh
cd ../docker/hbdb
./build_docker.sh
cd ../../scripts
./run_benchmark.sh
```

### Re-build the xclbin

I provide the xclbin in ``accelerator/bin``. If you want to re-build it, follow these steps.

Clone Xilinx's Vitis_Libraries from https://github.com/Xilinx/Vitis_Libraries and set XFLIB_DIR 
to Vitis_Libraries/security:

```
git clone https://github.com/Xilinx/Vitis_Libraries.git
export XFLIB_DIR=<path-to-Vitis_Libraries>/security
```

#### Keccak256

```
source <path-to-Vitis>/Vitis/2021.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

cd accelerator/hw/keccak256
make clean
faketime '2021-12-24 08:15:42' make build TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
make host TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
make run TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
```

#### ECDSA Verify

```
source <path-to-Vitis>/Vitis/2021.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

cd accelerator/hw/ecdsa_secp256k1
make clean
faketime '2021-12-24 08:15:42' make build TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
make host TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
make run TARGET=hw DEVICE=xilinx_u55n_gen3x4_xdma_1_202110_1
```

### Build the C++ driver (libkeccak.so)

```
cd accelerator/sw/keccak256
make lib
```

## License

Apache License Version 2.0
