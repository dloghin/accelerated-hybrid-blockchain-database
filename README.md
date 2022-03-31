# Accelerated Hybrid Blockchain Database

Accelerated Hybrid Blockchain Database

Branch "baseline2" - Keccak256 and ECDSA secp256k1 on CPU only (no FPGA acceleration)

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
./preprocess_ycsb_data.sh
cd ../docker/hbdb
./build_docker.sh
cd ../../scripts
./run_benchmark.sh
```

or, to run with different number of client threads:

```
./run_benchmark.sh all
```

