/*
 * Copyright 2022-2023 Dumitrel Loghin
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

/**
 * This is the Go-C interface for Keccak 256 and ECDSA secp256k1 signature verification
*/
#ifndef _HEADER_GPU_H_
#define _HEADER_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif
    void init_gpu_keccak(int batch_size);
    void init_gpu_secp256k1(int batch_size);
    void init_gpu(int batch_size);
    unsigned char* run_keccak(int batch_size, unsigned char* messages, int* message_lengths);
    unsigned char* run_secp256k1(int batch_size, unsigned char* pkeys, unsigned char* signatures, unsigned char* digests);
    unsigned char* run_keccak_secp256k1(int batch_size, unsigned char* pkeys, unsigned char* signatures, unsigned char* messages, int* message_lengths);
    void free_gpu_keccak();
    void free_gpu_secp256k1();
    void free_gpu();
#ifdef __cplusplus
}
#endif

#endif // _HEADER_GPU_H_