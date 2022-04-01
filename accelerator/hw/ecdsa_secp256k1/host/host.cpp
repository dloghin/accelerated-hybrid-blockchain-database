/**********
Copyright (c) 2021, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********/
/*
 * Modified by Dumitrel Loghin (2022)
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <iomanip>

#include <ap_int.h>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

// Each elements in all 5 input vectors has size 32 bytes
#define DIM		32

ap_uint<256> test_m = ap_uint<256>("0x44acf6b7e36c1342c2c5897204fe09504e1e2efb1a900377dbc4e7a6a133ec56");
ap_uint<256> test_r = ap_uint<256>("0xf3ac8061b514795b8843e3d6629527ed2afd6b1f6a555a7acabb5e6f79c8c2ac");
ap_uint<256> test_s = ap_uint<256>("0x8bf77819ca05a6b2786c76262bf7371cef97b218e96f175a3ccdda2acc058903");
ap_uint<256> test_Qx = ap_uint<256>("0x1ccbe91c075fc7f4f033bfa248db8fccd3565de94bbfb12f3c59ff46c271bf83");
ap_uint<256> test_Qy = ap_uint<256>("0xce4014c68811f9a21a1fdb2c0e6113e06db7ca93b7404e78dc7ccd5ca89a4ca9");

unsigned char hash[] = {
		0x44, 0xac, 0xf6, 0xb7, 0xe3, 0x6c, 0x13, 0x42,
		0xc2, 0xc5, 0x89, 0x72, 0x04, 0xfe, 0x09, 0x50,
		0x4e, 0x1e,	0x2e, 0xfb, 0x1a, 0x90, 0x03, 0x77,
		0xdb, 0xc4, 0xe7, 0xa6, 0xa1, 0x33, 0xec, 0x56
};
unsigned char r[] = {
		0xf3, 0xac, 0x80, 0x61, 0xb5, 0x14, 0x79, 0x5b,
		0x88, 0x43, 0xe3, 0xd6, 0x62, 0x95, 0x27, 0xed,
		0x2a, 0xfd, 0x6b, 0x1f, 0x6a, 0x55, 0x5a, 0x7a,
		0xca, 0xbb, 0x5e, 0x6f, 0x79, 0xc8, 0xc2, 0xac
};
unsigned char s[] = {
		0x8b, 0xf7, 0x78, 0x19, 0xca, 0x05, 0xa6, 0xb2,
		0x78, 0x6c, 0x76, 0x26, 0x2b, 0xf7, 0x37, 0x1c,
		0xef, 0x97, 0xb2, 0x18, 0xe9, 0x6f, 0x17, 0x5a,
		0x3c, 0xcd, 0xda, 0x2a, 0xcc, 0x05, 0x89, 0x03
};
unsigned char qx[] = {
		0x1c, 0xcb, 0xe9, 0x1c, 0x07, 0x5f, 0xc7, 0xf4,
		0xf0, 0x33, 0xbf, 0xa2, 0x48, 0xdb, 0x8f, 0xcc,
		0xd3, 0x56, 0x5d, 0xe9, 0x4b, 0xbf, 0xb1, 0x2f,
		0x3c, 0x59, 0xff, 0x46, 0xc2, 0x71, 0xbf, 0x83
};
unsigned char qy[] = {
		0xce, 0x40, 0x14, 0xc6, 0x88, 0x11, 0xf9, 0xa2,
		0x1a, 0x1f, 0xdb, 0x2c, 0x0e, 0x61, 0x13, 0xe0,
		0x6d, 0xb7, 0xca, 0x93, 0xb7, 0x40, 0x4e, 0x78,
		0xdc, 0x7c, 0xcd, 0x5c, 0xa8, 0x9a, 0x4c, 0xa9
};

void init_data(ap_uint<256>* buffer_hash,
		ap_uint<256>* buffer_qx, ap_uint<256>* buffer_qy,
		ap_uint<256>* buffer_r, ap_uint<256>* buffer_s,
		ap_uint<8>* outputs, unsigned int num_elements) {

	for (size_t i = 0; i < num_elements; i++) {
		buffer_hash[i] = test_m;
		buffer_qx[i] = test_Qx;
		buffer_qy[i] = test_Qy;
		buffer_r[i] = test_r;
		buffer_s[i] = test_s;
		outputs[i] = 0;
	}
}

void init_data_vector(unsigned char* buffer_hash,
		unsigned char* buffer_qx, unsigned char* buffer_qy,
		unsigned char* buffer_r, unsigned char* buffer_s,
		int* outputs, unsigned int num_elements) {

	for (size_t i = 0; i < num_elements; i++) {
		for (size_t j = 0; j < DIM; j++) {
			size_t idx = i * num_elements + j;
			buffer_hash[idx] = hash[j];
			buffer_qx[idx] = qx[j];
			buffer_qy[idx] = qy[j];
			buffer_r[idx] = r[j];
			buffer_s[idx] = s[j];
		}
		outputs[i] = 0;
	}
}

bool verify(ap_uint<8>* results, int num_elements) {
	bool match = true;
	for (int i = 0; i < num_elements; i++) {
		if (results[i] == 0) {
			match = false;
			break;
		}
	}
	std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
	return match;
}

int main(int argc, char **argv) {

	// Check input arguments
	if (argc < 2 || argc > 4) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File> <#elements(optional)> <debug(optional)>" << std::endl;
		return EXIT_FAILURE;
	}
	// Read FPGA binary file
	auto binaryFile = argv[1];
	unsigned int num_elements = 100;
	bool user_size = false;
	// Check if the user defined the # of elements
	if (argc >= 3){
		user_size = true;
		unsigned int val;
		try {
			val = std::stoi(argv[2]);
		}
		catch (const std::invalid_argument val) {
			std::cerr << "Invalid argument in position 2 (" << argv[2] << ") program expects an integer as number of elements" << std::endl;
			return EXIT_FAILURE;
		}
		catch (const std::out_of_range val) {
			std::cerr << "Number of elements out of range, try with a number lower than 2147483648" << std::endl;
			return EXIT_FAILURE;
		}
		num_elements = val;
		std::cout << "User number of elements enabled" << std::endl;
	}
	bool debug = false;
	// Check if the user defined debug
	if (argc == 4){
		std::string debug_arg = argv[3];
		if(debug_arg.compare("debug") == 0)
			debug = true;
		std::cout << "Debug enabled" << std::endl;
	}

	if (!user_size){
		// Define number of num_elements
		if (xcl::is_hw_emulation())
			num_elements= 100;
		else if (xcl::is_emulation())
			num_elements= 100;
		else{
			num_elements= 100;
		}
	}

	// I/O Data Vectors
	std::vector<ap_uint<256>, aligned_allocator<ap_uint<256>>> buffer_hash(DIM * num_elements);
	std::vector<ap_uint<256>, aligned_allocator<ap_uint<256>>> buffer_qx(DIM * num_elements);
	std::vector<ap_uint<256>, aligned_allocator<ap_uint<256>>> buffer_qy(DIM * num_elements);
	std::vector<ap_uint<256>, aligned_allocator<ap_uint<256>>> buffer_r(DIM * num_elements);
	std::vector<ap_uint<256>, aligned_allocator<ap_uint<256>>> buffer_s(DIM * num_elements);
	std::vector<ap_uint<8>, aligned_allocator<ap_uint<8>>> buffer_results(num_elements);

	// OpenCL Host Code Begins.
	// OpenCL objects
	cl::Device device;
	cl::Context context;
	cl::CommandQueue q;
	cl::Program program;
	cl::Kernel kernel;
	cl_int err;

	// get_xil_devices() is a utility API which will find the Xilinx
	// platforms and will return list of devices connected to Xilinx platform
	auto devices = xcl::get_xil_devices();

	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++) {
		device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
		OCL_CHECK(err,
				q = cl::CommandQueue(context, device,
						CL_QUEUE_PROFILING_ENABLE |
						CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
						&err));
		std::cout << "Trying to program device[" << i
											   << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		} else {
			std::cout << "Device[" << i << "]: program successful!\n";
			// Creating Kernel
			OCL_CHECK(err, kernel  = cl::Kernel(program, "verify_kernel" , &err));
			valid_device = true;
			break; // we break because we found a valid device
		}
	}
	if (!valid_device) {
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	std::cout << "Running secp256k1 Verify with " << num_elements << " elements" << std::endl;

	// Initialize the data vectors
	init_data(buffer_hash.data(), buffer_qx.data(), buffer_qy.data(), buffer_r.data(), buffer_s.data(), buffer_results.data(), num_elements);

	// Running the kernel
	unsigned int size_bytes  = num_elements * DIM;

	// Allocate Buffer in Global Memory
	// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
	// Device-to-host communication
	OCL_CHECK(err, cl::Buffer buffer_input_hash(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			size_bytes, buffer_hash.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input_qx(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			size_bytes, buffer_qx.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input_qy(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			size_bytes, buffer_qy.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input_r(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			size_bytes, buffer_r.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_input_s(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			size_bytes, buffer_s.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_output(context,
			CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
			size_bytes, buffer_results.data(), &err));


	// Setting Kernel Arguments krnl_vadd
	OCL_CHECK(err, err = kernel.setArg(0, buffer_input_hash));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_input_qx));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_input_qy));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_input_r));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_input_s));
	OCL_CHECK(err, err = kernel.setArg(5, buffer_output));
	OCL_CHECK(err, err = kernel.setArg(6, num_elements));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_hash, buffer_input_qx, buffer_input_qy, buffer_input_r, buffer_input_s}, 0 /* 0 means from host*/));
	OCL_CHECK(err, err = q.finish());

	// Launching the Kernels
	std::cout << "Launching Hardware Kernel..." << std::endl;
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	// wait for the kernel to finish their operations
	OCL_CHECK(err, err = q.finish());

	// Copy Result from Device Global Memory to Host Local Memory
	std::cout << "Getting Hardware Results..." << std::endl;
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
	OCL_CHECK(err, err = q.finish());

	// OpenCL Host Code Ends

	// Compare the device results with software results
	bool match = verify(buffer_results.data(), num_elements);

	return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
