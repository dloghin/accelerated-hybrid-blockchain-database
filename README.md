# Accelerated Hybrid Blockchain Database

Accelerated Hybrid Blockchain Database

Branch "baseline2" - ECDSA secp256k1 on CPU

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