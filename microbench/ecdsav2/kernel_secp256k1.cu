#include "ap_uint_cuda.h"

__device__ ap_uint<256> *a;  // = ap_uint<256>("0");
__device__ ap_uint<256> *p;  // = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
__device__ ap_uint<256> *n;  // = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
__device__ ap_uint<256> *Gx; // = ap_uint<256>("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
__device__ ap_uint<256> *Gy; // = ap_uint<256>("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

__global__ void init_gpu_env(cgbn_error_report_t *report, int batch_size)
{
    int tid = threadIdx.x;
    if (tid > 0)
        return;

    cgbn_env = new env_t*[batch_size];

    context_t bn_context(cgbn_report_monitor, report, tid); // construct a context
    env_t bn_env(bn_context.env<env_t>());                  // construct an environment
    cgbn_env[tid] = &bn_env;

    a = new ap_uint<256>("0");
    p = new ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    n = new ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    Gx = new ap_uint<256>("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    Gy = new ap_uint<256>("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
}

__global__ void destroy_gpu_env()
{
    int tid = threadIdx.x;
    if (tid > 0)
        return;

    delete a;
    delete p;
    delete n;
    delete Gx;
    delete Gy;
}

template <int N>
__device__ ap_uint<N + 1> productMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
    ap_uint<N + 1> tmp = 0;

    for (int i = N - 1; i >= 0; i--)
    {
        tmp <<= 1;
        if (tmp >= opM)
        {
            tmp -= opM;
        }
        if (opB[i] == 1)
        {
            tmp += opA;
            if (tmp >= opM)
            {
                tmp -= opM;
            }
        }
    }
    return tmp;
}

template <int N>
__device__ ap_uint<N> modularInv(ap_uint<N> opA, ap_uint<N> opM)
{
    // calc r = opA^-1 * 2^k and k
    ap_uint<N> u = opA;
    ap_uint<N> v = opM;
    ap_uint<N> x1 = ap_uint<N>(1);
    ap_uint<N> x2 = ap_uint<N>(0);

    while (u != 1 && v != 1)
    {
        while (u[0] == 0)
        {
            u = u >> 1;
            if (x1[0] == 0)
            {
                x1 = x1 >> 1;
            }
            else
            {
                x1 = (ap_uint<N + 1>(x1 + opM)) >> 1;
            }
        }
        while (v[0] == 0)
        {
            v = v >> 1;
            if (x2[0] == 0)
            {
                x2 = x2 >> 1;
            }
            else
            {
                x2 = (ap_uint<N + 1>(x2 + opM)) >> 1;
            }
        }
        if (u >= v)
        {
            u = u - v;
            if (x1 > x2)
            {
                x1 = x1 - x2;
            }
            else
            {
                x1 = (x1 - x2) + opM;
            }
        }
        else
        {
            v = v - u;
            if (x2 > x1)
            {
                x2 = x2 - x1;
            }
            else
            {
                x2 = (x2 - x1) + opM;
            }
        }
    }

    if (u == 1)
    {
        if (x1 > opM)
        {
            return x1 - opM;
        }
        else
        {
            return x1;
        }
    }
    else
    {
        if (x2 > opM)
        {
            return x2 - opM;
        }
        else
        {
            return x2;
        }
    }
}

template <int N>
__device__ ap_uint<N> subMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
    ap_uint<N + 1> opA1 = opA;
    ap_uint<N + 1> opB1 = opB;
    ap_uint<N + 1> opM1 = opM;
    ap_uint<N + 1> sum;
    if (opA >= opB)
    {
        sum = opA1 - opB1;
    }
    else
    {
        sum = opA1 + opM1;
        sum -= opB1;
    }
    return sum.range(N - 1, 0);    
}

template <int N>
__device__ ap_uint<N> addMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
    ap_uint<N + 1> opA1 = opA;
    ap_uint<N + 1> opB1 = opB;
    ap_uint<N + 1> sum = opA1 + opB1;
    if (sum >= opM) {
        sum -= opM;
    }
    return sum.range(N-1, 0);
}

__device__ ap_uint<256> productMod_p4(ap_uint<256> a, ap_uint<256> b)
{
    ap_uint<128> aH = a.range(255, 128);
    ap_uint<128> aL = a.range(127, 0);
    ap_uint<128> bH = b.range(255, 128);
    ap_uint<128> bL = b.range(127, 0);

    ap_uint<256> aLbH = aL * bH;
    ap_uint<256> aHbL = aH * bL;
    ap_uint<512> aHbH = aH * bH;
    ap_uint<256> aLbL = aL * bL;
    ap_uint<512> mid = aLbH + aHbL;

    ap_uint<512> mul = (aHbH << 256) + (mid << 128) + aLbL;
    ap_uint<256> P = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    ap_uint<256> c0 = mul.range(255, 0);
    ap_uint<256> c1 = mul.range(511, 256);
    ap_uint<256> w1 = 0;
    ap_uint<256> w2 = 0;
    ap_uint<256> w3 = 0;
    ap_uint<256> w4 = 0;
    ap_uint<256> w5 = 0;
    ap_uint<256> w6 = 0;

    w1.set_range(255, 32, c1.range(223, 0));
    w2.set_range(255, 9, c1.range(246, 0));
    w3.set_range(255, 8, c1.range(247, 0));
    w4.set_range(255, 7, c1.range(248, 0));
    w5.set_range(255, 6, c1.range(249, 0));
    w6.set_range(255, 4, c1.range(251, 0));

    ap_uint<256> s1 = c1.range(255, 252) + c1.range(255, 250) + c1.range(255, 249) + c1.range(255, 248) +
                      c1.range(255, 247) + c1.range(255, 224);
    
    ap_uint<256> k11 = (s1 << 2) + (s1 << 1) + s1;
    ap_uint<256> k = (s1 << 32) + (k11 << 7) + (s1 << 6) + (s1 << 4) + s1;

    ap_uint<256> tmp;
    tmp = addMod(k, c0, P);    
    tmp = addMod(tmp, w1, P);
    tmp = addMod(tmp, w2, P);
    tmp = addMod(tmp, w3, P);
    tmp = addMod(tmp, w4, P);
    tmp = addMod(tmp, w5, P);
    tmp = addMod(tmp, w6, P);
    tmp = addMod(tmp, c1, P);

    if (tmp >= P)
    {
        tmp -= P;
    }

    return tmp;
}

__device__ inline ap_uint<256> productMod_p(ap_uint<256> a, ap_uint<256> b) { return productMod_p4(a, b); }

__device__ inline ap_uint<256> productMod_n(ap_uint<256> a, ap_uint<256> b) { return productMod<256>(a, b, *n); }

__device__ void fromJacobian(ap_uint<256> X, ap_uint<256> Y, ap_uint<256> Z, ap_uint<256> &x, ap_uint<256> &y)
{
    if (Z == 0)
    {
        x = 0;
        y = 0;
    }
    else
    {
        ap_uint<256> ZInv = modularInv<256>(Z, *p);
        ap_uint<256> ZInv_2 = productMod_p(ZInv, ZInv);
        ap_uint<256> ZInv_3 = productMod_p(ZInv_2, ZInv);
        x = productMod_p(X, ZInv_2);
        y = productMod_p(Y, ZInv_3);
    }
}

__device__ void addJacobian(ap_uint<256> X1,
                            ap_uint<256> Y1,
                            ap_uint<256> Z1,
                            ap_uint<256> X2,
                            ap_uint<256> Y2,
                            ap_uint<256> Z2,
                            ap_uint<256> &X3,
                            ap_uint<256> &Y3,
                            ap_uint<256> &Z3)
{

    ap_uint<256> I1 = productMod_p(Z1, Z1);
    ap_uint<256> I2 = productMod_p(Z2, Z2);
    ap_uint<256> J1 = productMod_p(I1, Z1);
    ap_uint<256> J2 = productMod_p(I2, Z2);
    ap_uint<256> U1 = productMod_p(X1, I2);
    ap_uint<256> U2 = productMod_p(X2, I1);
    ap_uint<256> H = subMod<256>(U1, U2, *p);
    ap_uint<256> F = addMod<256>(H, H, *p);
    F = productMod_p(F, F);
    ap_uint<256> K1 = productMod_p(Y1, J2);
    ap_uint<256> K2 = productMod_p(Y2, J1);
    ap_uint<256> V = productMod_p(U1, F);
    ap_uint<256> G = productMod_p(F, H);
    ap_uint<256> R = subMod<256>(K1, K2, *p);
    R = addMod<256>(R, R, *p);

    if (Z2 == 0)
    {
        X3 = X1;
        Y3 = Y1;
        Z3 = Z1;
    }
    else if (Z1 == 0)
    {
        X3 = X2;
        Y3 = Y2;
        Z3 = Z2;
    }
    else if (addMod<256>(K1, K2, *p) == 0)
    {
        X3 = 1;
        Y3 = 1;
        Z3 = 0;
    }
    else
    {
        ap_uint<256> tmpX = productMod_p(R, R);
        ap_uint<256> tmp2V = addMod<256>(V, V, *p);
        tmpX = addMod<256>(tmpX, G, *p);
        X3 = subMod<256>(tmpX, tmp2V, *p);

        ap_uint<256> tmp2 = subMod<256>(V, X3, *p);
        tmp2 = productMod_p(tmp2, R);
        ap_uint<256> tmp4 = productMod_p(K1, G);
        tmp4 = addMod<256>(tmp4, tmp4, *p);
        Y3 = subMod<256>(tmp2, tmp4, *p);

        ap_uint<256> tmp5 = addMod<256>(Z1, Z2, *p);
        tmp5 = productMod_p(tmp5, tmp5);
        ap_uint<256> tmp6 = addMod<256>(I1, I2, *p);
        ap_uint<256> tmp7 = subMod<256>(tmp5, tmp6, *p);
        Z3 = productMod_p(tmp7, H);
    }
}

__device__ void doubleJacobian(
    ap_uint<256> X1, ap_uint<256> Y1, ap_uint<256> Z1, ap_uint<256> &X2, ap_uint<256> &Y2, ap_uint<256> &Z2)
{
    ap_uint<256> N = productMod_p(Z1, Z1);
    ap_uint<256> E = productMod_p(Y1, Y1);
    ap_uint<256> B = productMod_p(X1, X1);
    ap_uint<256> L = productMod_p(E, E);

    ap_uint<256> tmp1 = addMod<256>(X1, E, *p);
    tmp1 = productMod_p(tmp1, tmp1);
    ap_uint<256> tmp2 = addMod<256>(B, L, *p);
    ap_uint<256> tmp3 = subMod<256>(tmp1, tmp2, *p);
    ap_uint<256> S = addMod<256>(tmp3, tmp3, *p);

    ap_uint<256> tmp4 = productMod_p(N, N);
    tmp4 = productMod_p(tmp4, *a);
    ap_uint<256> tmp5 = addMod<256>(B, B, *p);
    tmp5 = addMod<256>(tmp5, B, *p);
    ap_uint<256> M = addMod<256>(tmp5, tmp4, *p);

    ap_uint<256> tmp6 = addMod<256>(S, S, *p);
    ap_uint<256> tmp7 = productMod_p(M, M);
    X2 = subMod<256>(tmp7, tmp6, *p);

    ap_uint<256> tmp8 = subMod<256>(S, X2, *p);
    tmp8 = productMod_p(tmp8, M);
    ap_uint<256> tmp9 = addMod<256>(L, L, *p);
    tmp9 = addMod<256>(tmp9, tmp9, *p);
    tmp9 = addMod<256>(tmp9, tmp9, *p);
    Y2 = subMod<256>(tmp8, tmp9, *p);

    ap_uint<256> tmp10 = addMod<256>(Y1, Z1, *p);
    tmp10 = productMod_p(tmp10, tmp10);
    ap_uint<256> tmp11 = addMod<256>(E, N, *p);
    Z2 = subMod<256>(tmp10, tmp11, *p);
}

__device__ void productJacobian(ap_uint<256> X1,
                                ap_uint<256> Y1,
                                ap_uint<256> Z1,
                                ap_uint<256> k,
                                ap_uint<256> &X2,
                                ap_uint<256> &Y2,
                                ap_uint<256> &Z2)
{

    ap_uint<256> RX = 1;
    ap_uint<256> RY = 1;
    ap_uint<256> RZ = 0;

    ap_uint<256> tmpX, tmpY, tmpZ, tmpRX, tmpRY, tmpRZ;

    for (int i = 0; i < 256; i++)
    {        
        addJacobian(RX, RY, RZ, X1, Y1, Z1, tmpRX, tmpRY, tmpRZ);
        doubleJacobian(X1, Y1, Z1, tmpX, tmpY, tmpZ);

        RX = (k[i] == 1) ? tmpRX : RX;
        RY = (k[i] == 1) ? tmpRY : RY;
        RZ = (k[i] == 1) ? tmpRZ : RZ;

        X1 = tmpX;
        Y1 = tmpY;
        Z1 = tmpZ;
    }

    X2 = RX;
    Y2 = RY;
    Z2 = RZ;
}

__device__ int verify(ap_uint<256> r, ap_uint<256> s, ap_uint<256> hash, ap_uint<256> Px, ap_uint<256> Py)
{
    if (r == 0 || r >= *n || s == 0 || s >= *n)
    {
        return 0;
    }
    else
    {
        ap_uint<256> z;
        z = hash;

        if (z >= *n)
        {
            z -= *n;
        }

        ap_uint<256> sInv = modularInv<256>(s, *n);

        ap_uint<256> u1 = productMod_n(sInv, z);
        ap_uint<256> u2 = productMod_n(sInv, r);

        ap_uint<256> x, y;

        ap_uint<256> t1x, t1y, t1z, t2x, t2y, t2z;
        ap_uint<256> t3x, t3y, t3z;

        productJacobian(*Gx, *Gy, ap_uint<256>(1), u1, t1x, t1y, t1z);
        productJacobian(Px, Py, ap_uint<256>(1), u2, t2x, t2y, t2z);

        addJacobian(t1x,
                    t1y,
                    t1z,
                    t2x,
                    t2y,
                    t2z,
                    t3x,
                    t3y,
                    t3z);

        fromJacobian(t3x, t3y, t3z, x, y);        

        if (x == 0 && y == 0)
        {
            return 0;
        }
        else
        {
            if (r == x)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
    }
}

__global__ void verify_batch(cgbn_error_report_t *report, packet_t *data, int *res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    context_t bn_context(cgbn_report_monitor, report, tid); // construct a context
    env_t bn_env(bn_context.env<env_t>());                  // construct an environment
    cgbn_env[tid] = &bn_env;

    assert(a != NULL);
    assert(p != NULL);
    assert(n != NULL);
    assert(Gx != NULL);
    assert(Gy != NULL);

    ap_uint<256> r = ap_uint<256>(&data[tid].r);
    ap_uint<256> s = ap_uint<256>(&data[tid].s);
    ap_uint<256> h = ap_uint<256>(&data[tid].h);
    ap_uint<256> px = ap_uint<256>(&data[tid].px);
    ap_uint<256> py = ap_uint<256>(&data[tid].py);

    res[tid] = verify(r, s, h, px, py);    
}

__global__ void verify_batch_v2(cgbn_error_report_t *report, unsigned char* pkeys, unsigned char* digests, unsigned char* signatures, unsigned char* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    context_t bn_context(cgbn_report_monitor, report, tid); // construct a context
    env_t bn_env(bn_context.env<env_t>());                  // construct an environment
    cgbn_env[tid] = &bn_env;

    assert(a != NULL);
    assert(p != NULL);
    assert(n != NULL);
    assert(Gx != NULL);
    assert(Gy != NULL);

    ap_uint<256> r = ap_uint<256>(&signatures[tid * 64], 32);
    ap_uint<256> s = ap_uint<256>(&signatures[tid * 64 + 32], 32);
    ap_uint<256> h = ap_uint<256>(&digests[tid * 32], 32);
    ap_uint<256> px = ap_uint<256>(&pkeys[tid * 64], 32);
    ap_uint<256> py = ap_uint<256>(&pkeys[tid * 64 + 32], 32);

    res[tid] = (unsigned char)verify(r, s, h, px, py);    
}

#ifdef GO_LIB

#include "secp256k1_gpu.h"

unsigned char *gpu_pkeys, *gpu_digests, *gpu_signatures, *gpu_res;
cgbn_error_report_t *gpu_report;
unsigned char *cpu_res;

void init_gpu(int batch_size)
{
    cpu_res = (unsigned char *)malloc(batch_size);

    CHECKCUDAERR(cgbn_error_report_alloc(&gpu_report));
    CHECKCUDAERR(cudaMalloc(&gpu_pkeys, 64 * batch_size));
    CHECKCUDAERR(cudaMalloc(&gpu_digests, 32 * batch_size));
    CHECKCUDAERR(cudaMalloc(&gpu_signatures, 64 * batch_size));
    CHECKCUDAERR(cudaMalloc(&gpu_res, batch_size));    

    init_gpu_env<<<1, 1>>>(gpu_report, batch_size);
    // CHECKCUDAERR(cudaGetLastError());
}

unsigned char* run_kernel(int batch_size, unsigned char* pkeys, unsigned char* digests, unsigned char* signatures)
{
    // transfer CPU -> GPU
    CHECKCUDAERR(cudaMemcpy(gpu_pkeys, pkeys, 64 * batch_size, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(gpu_digests, digests, 32 * batch_size, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(gpu_signatures, signatures, 64 * batch_size, cudaMemcpyHostToDevice));
    
    verify_batch_v2<<<batch_size/TPB, TPB>>>(gpu_report, gpu_pkeys, gpu_digests, gpu_signatures, gpu_res);
    // CHECKCUDAERR(cudaGetLastError());
    
    // transfer GPU -> CPU
    CHECKCUDAERR(cudaMemcpy(cpu_res, gpu_res, batch_size, cudaMemcpyDeviceToHost));    

    return cpu_res;
}

void free_gpu() 
{
    destroy_gpu_env<<<1, 1>>>();
}

#endif

#ifdef TEST

int main()
{
    char r[65] = "9852734efff86c8d38a71d3d5f33e5f6804a5f98594db4b2ab7a6a48651f7022";    
    char s[65] = "7576aed0b12ea1eab03f8d4c39c59ac3c213e20cb9381a3144b37949c151e170";
    char h1[65] = "9230175b13981da14d2f3334f321eb78fa0473133f6da3de896feb22fb258936";
    char h2[65] = "38f300a5a17105effbfd75c0a13e3ad2a4d65353eaf464df64fe4d3b2186aef9";
    char px[65] = "c7b15d2bdf22b4351ecd30e7eedf120124496a68b2280018ca817d1786f191b4";
    char py[65] = "b5b5025f28b9dc46339f0c342dceec5f7f36e3ef77e0336657d5c215975368fe";

    // CPU structures
    cudaError_t cpu_err;
    int *cpu_res = (int *)malloc(BATCH * sizeof(int));
    packet_t cpu_data1;
    packet_t cpu_data2;
    str_to_cgbn_mem_t(&cpu_data1.r, r);
    str_to_cgbn_mem_t(&cpu_data1.s, s);
    str_to_cgbn_mem_t(&cpu_data1.h, h1);
    str_to_cgbn_mem_t(&cpu_data1.px, px);
    str_to_cgbn_mem_t(&cpu_data1.py, py);
    str_to_cgbn_mem_t(&cpu_data2.r, r);
    str_to_cgbn_mem_t(&cpu_data2.s, s);
    str_to_cgbn_mem_t(&cpu_data2.h, h2);
    str_to_cgbn_mem_t(&cpu_data2.px, px);
    str_to_cgbn_mem_t(&cpu_data2.py, py);    

    print_bn(&cpu_data1.r);
    print_bn(&cpu_data1.s);
    print_bn(&cpu_data1.h);
    print_bn(&cpu_data1.px);
    print_bn(&cpu_data1.py);

    // GPU structures
    packet_t *gpu_data;
    int *gpu_res;
    cgbn_error_report_t *report;
    cudaError_t *gpu_err;    

    CHECKCUDAERR(cgbn_error_report_alloc(&report));
    CHECKCUDAERR(cudaMalloc(&gpu_data, BATCH * sizeof(packet_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_res, BATCH * sizeof(int)));
    CHECKCUDAERR(cudaMalloc(&gpu_err, sizeof(cudaError_t)));    

    for (int i = 0; i < BATCH; i++)
    {
        if (i % 2 == 0)
        {
            CHECKCUDAERR(cudaMemcpy(gpu_data + i, &cpu_data1, sizeof(packet_t), cudaMemcpyHostToDevice));
        }
        else
        {
            CHECKCUDAERR(cudaMemcpy(gpu_data + i, &cpu_data2, sizeof(packet_t), cudaMemcpyHostToDevice));
        }
    }

    init_gpu_env<<<1, 1>>>(report, BATCH);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaMemcpy(&cpu_err, gpu_res, sizeof(cudaError_t), cudaMemcpyDeviceToHost));
    if (cpu_err != cudaSuccess)
    {
        printf("Err: %d\n", cpu_err);
    }
    verify_batch<<<BATCH/TPB, TPB>>>(report, gpu_data, gpu_res);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaMemcpy(cpu_res, gpu_res, BATCH * sizeof(int), cudaMemcpyDeviceToHost));     

    for (int i = 0; i < BATCH; i++)
    {
        printf("%d ", cpu_res[i]);
    }
    printf("\n");

    destroy_gpu_env<<<1, 1>>>();

    CHECKCUDAERR(cudaFree(gpu_data));
    CHECKCUDAERR(cudaFree(gpu_res));
    CHECKCUDAERR(cudaFree(gpu_err));
    free(cpu_res);
}
#endif
