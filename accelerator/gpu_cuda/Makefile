CXX = g++
NVCC = nvcc
CUDA_ARCH = sm_80

kernel-secp256k1-cpu: kernel_secp256k1.cpp
	rm -f kernel-cpu
	$(CXX) -g -DTEST $^ -o $@ -lgmp

kernel-secp256k1-gpu: kernel_secp256k1.cu
	$(NVCC) -DTEST -I$$HOME/git/CGBN/include -arch=$(CUDA_ARCH) $^ -o $@ -lgmp

libacc-gpu.so: kernel_secp256k1.cu kernel_sha3.cu
	$(NVCC) -DGO_LIB --compiler-options '-fPIC' --shared -I$$HOME/git/CGBN/include -arch=$(CUDA_ARCH) $^ -o $@ -lgmp

# Debug
# $(NVCC) -I$$HOME/git/CGBN/include -arch=$(CUDA_ARCH) -lineinfo $^ -o $@ -lgmp

lib: libacc-gpu.so

clean:
	rm -f kernel-cpu
