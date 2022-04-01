/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 /*
  * Modified by Dumitrel Loghin (2022)
  */
#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <new>
#include <algorithm>
#include <cstdlib>
#include <iterator>

#include <sys/time.h>
#include "xf_utils_sw/logger.hpp"

inline int tvdiff(struct timeval* tv0, struct timeval* tv1) {
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = NULL;

    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();

    return reinterpret_cast<T*>(ptr);
}

class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

const char message[] = "abcdefghijklmnopqrstuvwxyz000000";

int main(int argc, const char* argv[]) {
    int nerr = 0;
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    xf::common::utils_sw::Logger logger;

    int num = 4;    
    int len8 = 32;    
    ap_uint<128>* msgLen = aligned_alloc<ap_uint<128> >(num);
    ap_uint<256>* results = aligned_alloc<ap_uint<256> >(num);    
    ap_uint<64>* msg = aligned_alloc<ap_uint<64> >(num * len8);
    
    int offidx = 0;
    for (int i = 0; i < num; i++) {
        printf("Offset idx: %d\n", offidx);
        results[i] = 0;
        msgLen[i].range(127, 64) = (uint64_t) 0;
        msgLen[i].range(63, 0) = (uint64_t) 4;
        msg[offidx] = 0;
        for (int j = 0; j < 4; j++) {
            int idx2 = (j % 4) * 8;
            msg[offidx].range(idx2 + 7, idx2) = (unsigned)message[j];            
            if (j % 4 == 3) {
                offidx++;
                msg[offidx] = 0;
            }
        }
        // if (i % 4 != 3) offidx++;
    }

    // do pre-process on CPU
    struct timeval start_time, end_time;
    // platform related operations
    cl_int err = CL_SUCCESS;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &err);

    // cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);

    cl::Program program(context, devices, xclBins, NULL, &err);

    cl::Kernel kernel(program, "keccak256_kernel", &err);
    logger.logCreateKernel(err);

    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[3];
    mext_o[0] = {2, msg, kernel()};
    mext_o[1] = {3, msgLen, kernel()};
    mext_o[2] = {4, results, kernel()};

    // create device buffer and map dev buf to host buf
    cl::Buffer msg_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                    sizeof(ap_uint<64>) * num * len8, &mext_o[0]);
    cl::Buffer msgLen_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       sizeof(ap_uint<128>) * num, &mext_o[1]);
    cl::Buffer digest_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                       sizeof(ap_uint<256>) * num, &mext_o[2]);

    kernel.setArg(0, num);
    kernel.setArg(1, offidx);
    kernel.setArg(2, msg_buf);
    kernel.setArg(3, msgLen_buf);
    kernel.setArg(4, digest_buf);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Memory> ob_init;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_init.resize(0);
    ob_init.push_back(msg_buf);
    ob_init.push_back(msgLen_buf);

    q.enqueueMigrateMemObjects(ob_init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    ob_in.resize(0);
    ob_in.push_back(msg_buf);
    ob_in.push_back(msgLen_buf);
    ob_out.resize(0);
    ob_out.push_back(digest_buf);

    events_write.resize(1);
    events_kernel.resize(1);
    events_read.resize(1);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.enqueueTask(kernel, &events_write, &events_kernel[0]);
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;

    unsigned long time1, time2, total_time;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    total_time = time2 - time1;
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Kernel Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    total_time += time2 - time1;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    total_time = time2 - time1;
    std::cout << "Total Execution time " << total_time / 1000000.0 << " ms" << std::endl;

    unsigned char res;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < 32; j++) {
            res = results[i].range(8 * j + 7, 8 * j);
            printf("%x", res);
        }
        printf("\n");
    }
    /*
    uint64_t res[4];
    memcpy(res, results, 32);
    printf("%lx %lx %lx %lx\n", res[0], res[1], res[2], res[3]);
    */

    return nerr;
}
