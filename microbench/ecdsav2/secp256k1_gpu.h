/*
 * Copyright 2022 Dumitrel Loghin
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
 * This is the Go-C interface for ECDSA secp256k1 signature verification
*/
#ifdef __cplusplus
extern "C" {
#endif
    void init_gpu(int batch_size);    
    unsigned char* run_kernel(int batch_size, unsigned char* pkeys, unsigned char* digests, unsigned char* signatures);    
    void free_gpu();
#ifdef __cplusplus
}
#endif
