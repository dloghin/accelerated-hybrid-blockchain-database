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

cd ..
mkdir -p bin
cd bin

set -x

go version

go build -o preprocess ../cmd/preprocess/main.go
go build -o crypto ../cmd/crypto/main.go
go build -o hbdb-server ../cmd/hbdb/main.go
go build -o hbdb-bench ../src/benchmark/ycsbbench/main.go
go build -o hbdb-batch-bench ../src/benchmark/ycsbbenchbatch/main.go
