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
EXTPATH = ../../ext
CXX = g++
CXXFLAGS += -g -I$(EXTPATH) -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++14 -O0 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE

all: keccak.exe

lib: libkeccak.so

keccak.exe: keccak.cpp keccak.hpp $(EXTPATH)/xcl2.cpp test_keccak.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

libkeccak.so: keccak.cpp keccak.hpp keccak.h $(EXTPATH)/xcl2.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared $^ -o $@ $(LDFLAGS)

run: keccak.exe
	taskset -c 0 ./keccak.exe -xclbin ../../bin/keccak256_kernel.xclbin -reqpath ../scripts/requests.txt
