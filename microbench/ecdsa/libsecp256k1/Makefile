CC = nvcc

# CFLAGS = -DUSE_SCALAR_8X32 -DUSE_FIELD_10X26 -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM -DUSE_SCALAR_INV_NUM
CFLAGS = -DUSE_CUDA -DUSE_NUM_NONE -DUSE_FIELD_INV_BUILTIN -DUSE_SCALAR_INV_BUILTIN -DUSE_FIELD_10X26 -DUSE_SCALAR_8X32

SRCS=$(wildcard *.c)

OBJS=$(SRCS:.c=.o)

all: $(SRCS)
	$(CC) $^ $(CFLAGS) -c

secp256k1: secp256k1.cu
	$(CC) $^ $(CFLAGS) -c

sign: bench_sign.o secp256k1.o
	$(CC) $^ -o $@

verify-gpu: bench_verify.cu secp256k1.cu
	$(CC) $^ $(CFLAGS) -o $@
