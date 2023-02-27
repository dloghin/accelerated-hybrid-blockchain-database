#ifndef _SECP256K1_CONTEXT_H_
#define _SECP256K1_CONTEXT_H_

#include "secp256k1.h"
#include "ecmult.h"
#include "ecmult_gen.h"

struct secp256k1_context_struct {
    secp256k1_ecmult_context ecmult_ctx;
    secp256k1_ecmult_gen_context ecmult_gen_ctx;
    secp256k1_callback illegal_callback;
    secp256k1_callback error_callback;
};

#endif