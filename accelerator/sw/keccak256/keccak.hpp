#pragma once

#include "xcl2.hpp"
#include <CL/cl_ext_xilinx.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <new>
#include <algorithm>
#include <cstdlib>
#include <iterator>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BATCH_SIZE 100
#define MAX_DIM 256

typedef struct {
    cl::Kernel kernel;
    cl::CommandQueue queue;
    cl::Buffer msg_buf;
    cl::Buffer msgLen_buf;
    cl::Buffer digest_buf;
} KernelQueue;

extern ap_uint<128> *msgLen;
extern ap_uint<256> *results;
extern ap_uint<64> *msg;

KernelQueue init(std::string xclbin_path, int num);
void run(KernelQueue kq, int num, int size);

int read_data(FILE *fin, int num);
void write_data(FILE *fout, int num);