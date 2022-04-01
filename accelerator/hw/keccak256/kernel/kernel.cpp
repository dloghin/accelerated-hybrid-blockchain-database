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

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/keccak256.hpp"

void readLenM2S(int n,
		ap_uint<128>* msgLen,
		hls::stream<ap_uint<128>>& msgLenStrm,
		hls::stream<bool>& endMsgLenStrm) {
	
	// n is the size of msgLen array (== num)
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
    	msgLenStrm.write(msgLen[i]);
    	endMsgLenStrm.write(0);
    }
    endMsgLenStrm.write(1);
}

void readDataM2S(int n, 
		ap_uint<64>* msg,
		hls::stream<ap_uint<64>>& msgStrm) {

	// n is the size of msg array (== size)
	for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
		msgStrm.write(msg[i]);		
	}	
}

void writeS2M(int n, hls::stream<ap_uint<256>>& digestStrm, hls::stream<bool>& endDigestStrm, ap_uint<256>* digest) {
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
    	endDigestStrm.read();
        digest[i] = digestStrm.read();
    }
    endDigestStrm.read();
}

/*
 * num - input size (number of elements)
 * msg - input message
 * msgLen - input message lengths (elements)
 * digest - output (32 bytes = 256 bits per digest)
 */
extern "C" void keccak256_kernel(
	int num, int size, 
	ap_uint<64>* msg, ap_uint<128>* msgLen, ap_uint<256>* digest) {

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \    
    bundle = gmem0 port = msg
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \    
    bundle = gmem1 port = msgLen
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \    
    bundle = gmem1 port = digest

#pragma HLS INTERFACE s_axilite port = num bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = msg bundle = control
#pragma HLS INTERFACE s_axilite port = msgLen bundle = control
#pragma HLS INTERFACE s_axilite port = digest bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS dataflow

	hls::stream<ap_uint<64> > msgStrm("s1");
	hls::stream<ap_uint<128> > msgLenStrm("s2");
	hls::stream<bool> endMsgLenStrm("s3");
	hls::stream<ap_uint<256> > digestStrm("s4");
	hls::stream<bool> endDigestStrm("s5");
	// hls::stream<bool> endMsgCopyStrm("s6");

#pragma HLS stream variable = msgStrm depth = 16384
#pragma HLS stream variable = msgLenStrm depth = 128
#pragma HLS stream variable = digestStrm depth = 128
#pragma HLS stream variable = endMsgLenStrm depth = 128
#pragma HLS stream variable = endDigestStrm depth = 128

    readDataM2S(size, msg, msgStrm);
	readLenM2S(num, msgLen, msgLenStrm, endMsgLenStrm);
    xf::security::keccak_256(msgStrm, msgLenStrm, endMsgLenStrm, digestStrm, endDigestStrm);
    writeS2M(num, digestStrm, endDigestStrm, digest);
}
