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

#include "xf_security/ecdsa_secp256k1.hpp"

extern "C" {

const ap_uint<256> r = ap_uint<256>("0xf3ac8061b514795b8843e3d6629527ed2afd6b1f6a555a7acabb5e6f79c8c2ac");
const ap_uint<256> s = ap_uint<256>("0x8bf77819ca05a6b2786c76262bf7371cef97b218e96f175a3ccdda2acc058903");
const ap_uint<256> Qx = ap_uint<256>("0x1ccbe91c075fc7f4f033bfa248db8fccd3565de94bbfb12f3c59ff46c271bf83");
const ap_uint<256> Qy = ap_uint<256>("0xce4014c68811f9a21a1fdb2c0e6113e06db7ca93b7404e78dc7ccd5ca89a4ca9");

void verify_kernel(const ap_uint<256>* hash,		
		ap_uint<8>* results,       	// Output Results
		int elements     			// Number of elements
) {
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 16 bundle = gmem0 port = hash
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    1 max_write_burst_length = 16 max_read_burst_length = 2 bundle = gmem1 port = results
#pragma HLS INTERFACE s_axilite port = elements bundle = control
#pragma HLS INTERFACE s_axilite port = hash bundle = control
#pragma HLS INTERFACE s_axilite port = results bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
	
	xf::security::ecdsaSecp256k1<256> processor;
    processor.init();
	for (int i = 0; i < elements; i++) {
// #pragma HLS LOOP_TRIPCOUNT avg=100 max=100 min=100
#pragma HLS UNROLL factor=10
		results[i] = processor.verify(r, s, hash[i], Qx, Qy);
	}
}
}
