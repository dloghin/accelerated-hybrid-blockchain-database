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
