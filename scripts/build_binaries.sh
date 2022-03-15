#!/bin/bash

cd ..
mkdir -p bin
cd bin

set -x

go version

go build -o preprocess ../cmd/preprocess/main.go
go build -o crypto ../cmd/crypto/main.go
go build -o hbdb-server ../cmd/hbdb/main.go
go build -o hbdb-bench ../src/benchmark/ycsbbench/main.go
