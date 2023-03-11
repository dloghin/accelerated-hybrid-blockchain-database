CXX = g++
NVCC = nvcc

kernel-cpu: kernel_secp256k1.cpp
	rm -f kernel-cpu
	$(CXX) -g -DTEST $^ -o $@ -lgmp

kernel-gpu: kernel_secp256k1.cu
	$(NVCC) -DTEST -I/home/elwing/git/CGBN/include -arch=sm_80 $^ -o $@ -lgmp

libsecp256k1-gpu.so: kernel_secp256k1.cu
	$(NVCC) -DGO_LIB --compiler-options '-fPIC' --shared -I/home/elwing/git/CGBN/include -arch=sm_80 $^ -o $@ -lgmp

# Debug
# $(NVCC) -I/home/dumi/git/CGBN/include -arch=sm_50 -lineinfo $^ -o $@ -lgmp

clean:
	rm -f kernel-cpu
