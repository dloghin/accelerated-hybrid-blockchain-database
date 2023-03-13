CXX = g++
NVCC = nvcc

kernel-secp256k1-cpu: kernel_secp256k1.cpp
	rm -f kernel-cpu
	$(CXX) -g -DTEST $^ -o $@ -lgmp

kernel-secp256k1-gpu: kernel_secp256k1.cu
	$(NVCC) -DTEST -I$$HOME/git/CGBN/include -arch=sm_50 $^ -o $@ -lgmp

libacc-gpu.so: kernel_secp256k1.cu kernel_sha3.cu
	$(NVCC) -DGO_LIB --compiler-options '-fPIC' --shared -I$$HOME/git/CGBN/include -arch=sm_50 $^ -o $@ -lgmp

# Debug
# $(NVCC) -I$$HOME/git/CGBN/include -arch=sm_50 -lineinfo $^ -o $@ -lgmp

lib: libacc-gpu.so

clean:
	rm -f kernel-cpu
