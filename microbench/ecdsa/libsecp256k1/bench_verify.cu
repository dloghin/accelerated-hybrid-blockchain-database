/**********************************************************************
 * Copyright (c) 2014 Pieter Wuille                                   *
 * Distributed under the MIT software license, see the accompanying   *
 * file COPYING or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#include <stdio.h>
#include <string.h>

#include "secp256k1.h"
#include "util.h"
#include "bench.h"
#include "context.h"

#ifdef ENABLE_OPENSSL_TESTS
#include <openssl/bn.h>
#include <openssl/ecdsa.h>
#include <openssl/obj_mac.h>
#endif

typedef struct {
    secp256k1_context *ctx;
    unsigned char msg[32];
    unsigned char key[32];
    unsigned char sig[72];
    size_t siglen;
    unsigned char pubkey[33];
    size_t pubkeylen;
#ifdef ENABLE_OPENSSL_TESTS
    EC_GROUP* ec_group;
#endif
} benchmark_verify_t;

static void benchmark_verify(void* arg) {
    int i;
    benchmark_verify_t* data = (benchmark_verify_t*)arg;

    for (i = 0; i < 20000; i++) {
        secp256k1_pubkey pubkey;
        secp256k1_ecdsa_signature sig;
        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
        CHECK(secp256k1_ec_pubkey_parse(data->ctx, &pubkey, data->pubkey, data->pubkeylen) == 1);
        CHECK(secp256k1_ecdsa_signature_parse_der(data->ctx, &sig, data->sig, data->siglen) == 1);
        CHECK(secp256k1_ecdsa_verify(data->ctx, &sig, data->msg, &pubkey) == (i == 0));
        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
    }
}

#define BATCH 128

static void benchmark_verify_gpu(void* arg) {
    int i, k;
    benchmark_verify_t* data = (benchmark_verify_t*)arg;

    secp256k1_context** gpu_ctx;
    secp256k1_ecdsa_signature* gpu_sig;
    secp256k1_pubkey* gpu_pubkey;
    unsigned char** gpu_msg;
    int* gpu_res;
    int* cpu_res;
    cudaMalloc(&gpu_ctx, BATCH * sizeof(secp256k1_context*));
    cudaMalloc(&gpu_sig, BATCH * sizeof(secp256k1_ecdsa_signature));
    cudaMalloc(&gpu_pubkey, BATCH * sizeof(secp256k1_pubkey));
    cudaMalloc(&gpu_msg, BATCH * sizeof(unsigned char*));
    cudaMalloc(&gpu_res, BATCH * sizeof(int));
    for (i = 0; i < BATCH; i++) {
        cudaMalloc(&gpu_ctx[i], sizeof(secp256k1_context));
        cudaMalloc(&gpu_msg[i], 32);
    }
    cpu_res = (int*)malloc(BATCH * sizeof(int));

    for (k = 0, i = 0; i < 12800; k++, i++) {
        if (k == BATCH) {
            secp256k1_ecdsa_verify_gpu<<<1, BATCH>>>(gpu_ctx, gpu_sig, gpu_msg, gpu_pubkey, gpu_res);
            cudaMemcpy(gpu_res, cpu_res, 32, cudaMemcpyDeviceToHost);
            k = 0;
        }
        secp256k1_pubkey pubkey;
        secp256k1_ecdsa_signature sig;
        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
        CHECK(secp256k1_ec_pubkey_parse(data->ctx, &pubkey, data->pubkey, data->pubkeylen) == 1);
        CHECK(secp256k1_ecdsa_signature_parse_der(data->ctx, &sig, data->sig, data->siglen) == 1);
        // CHECK(secp256k1_ecdsa_verify(data->ctx, &sig, data->msg, &pubkey) == (i == 0));

        cudaMemcpy(gpu_ctx[k], data->ctx, sizeof(secp256k1_context), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_sig + k, &sig, sizeof(secp256k1_ecdsa_signature), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_pubkey + k, &pubkey, sizeof(secp256k1_pubkey), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_msg[k], data->msg, 32, cudaMemcpyHostToDevice);

        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
    }

    for (i = 0; i < BATCH; i++) {
        cudaFree(gpu_ctx[i]);
        cudaFree(gpu_msg[i]);
    }
    cudaFree(gpu_ctx);
    cudaFree(gpu_sig);
    cudaFree(gpu_pubkey);
    cudaFree(gpu_msg);
    cudaFree(gpu_res);
    free(cpu_res);
}

#ifdef ENABLE_OPENSSL_TESTS
static void benchmark_verify_openssl(void* arg) {
    int i;
    benchmark_verify_t* data = (benchmark_verify_t*)arg;

    for (i = 0; i < 20000; i++) {
        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
        {
            EC_KEY *pkey = EC_KEY_new();
            const unsigned char *pubkey = &data->pubkey[0];
            int result;

            CHECK(pkey != NULL);
            result = EC_KEY_set_group(pkey, data->ec_group);
            CHECK(result);
            result = (o2i_ECPublicKey(&pkey, &pubkey, data->pubkeylen)) != NULL;
            CHECK(result);
            result = ECDSA_verify(0, &data->msg[0], sizeof(data->msg), &data->sig[0], data->siglen, pkey) == (i == 0);
            CHECK(result);
            EC_KEY_free(pkey);
        }
        data->sig[data->siglen - 1] ^= (i & 0xFF);
        data->sig[data->siglen - 2] ^= ((i >> 8) & 0xFF);
        data->sig[data->siglen - 3] ^= ((i >> 16) & 0xFF);
    }
}
#endif

int main(void) {
    int i;
    secp256k1_pubkey pubkey;
    secp256k1_ecdsa_signature sig;
    benchmark_verify_t data;

    data.ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    for (i = 0; i < 32; i++) {
        data.msg[i] = 1 + i;
    }
    for (i = 0; i < 32; i++) {
        data.key[i] = 33 + i;
    }
    data.siglen = 72;
    CHECK(secp256k1_ecdsa_sign(data.ctx, &sig, data.msg, data.key, NULL, NULL));
    CHECK(secp256k1_ecdsa_signature_serialize_der(data.ctx, data.sig, &data.siglen, &sig));
    CHECK(secp256k1_ec_pubkey_create(data.ctx, &pubkey, data.key));
    data.pubkeylen = 33;
    CHECK(secp256k1_ec_pubkey_serialize(data.ctx, data.pubkey, &data.pubkeylen, &pubkey, SECP256K1_EC_COMPRESSED) == 1);

    run_benchmark("ecdsa_verify", benchmark_verify_gpu, NULL, NULL, &data, 10, 20000);
#ifdef ENABLE_OPENSSL_TESTS
    data.ec_group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    run_benchmark("ecdsa_verify_openssl", benchmark_verify_openssl, NULL, NULL, &data, 10, 20000);
    EC_GROUP_free(data.ec_group);
#endif

    secp256k1_context_destroy(data.ctx);
    return 0;
}
