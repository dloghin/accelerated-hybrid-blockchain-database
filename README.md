# Accelerated Hybrid Blockchain Database

Accelerated Hybrid Blockchain Database

## Getting started

### Pre-requisites

- Install dependencies such as Docker, go, Java (for YCSB)
- Download and prepare YCSB data
- Build binaries
- Build docker images

```
cd scripts
./install_dependencies.sh
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

I provide the xclbin in ``accelerator/bin``. If you want to re-build it, follow this steps:

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
