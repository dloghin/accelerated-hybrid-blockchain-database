#include <cuda.h>
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <assert.h>

#define TPI 1
#define BITS 512
#define BATCH 128

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

typedef struct
{
    cgbn_mem_t<BITS> r;
    cgbn_mem_t<BITS> s;
    cgbn_mem_t<BITS> h;
    cgbn_mem_t<BITS> px;
    cgbn_mem_t<BITS> py;
} packet_t;

static const unsigned char cpu_hex2str[6] = {0xA, 0xB, 0xC, 0xD, 0xE, 0xF};
__device__ __constant__ unsigned char gpu_hex2str[6] = {0xA, 0xB, 0xC, 0xD, 0xE, 0xF};

__host__ __device__ void str_to_cgbn_mem_t(cgbn_mem_t<BITS> *addr, const char *str)
{
    const unsigned char *hex2str = NULL;
#ifdef __CUDA_ARCH__
    hex2str = gpu_hex2str;
#else
    hex2str = cpu_hex2str;
#endif

    char *ptr = (char *)str;
    while (*ptr != '\0')
    {
        ptr++;
    }
    int l = ptr - str;  // string size   
    int sht = 0; // shift factor
    for (int i = 0; i < sizeof(addr->_limbs) / sizeof(addr->_limbs[0]); i++) {
        addr->_limbs[i] = 0;
    }    
    for (int j = 0, i = l - 1; i >= 0; i--, j++)
    {
        uint32_t v = (uint32_t)(str[i] - '0');
        if (str[i] >= 'a' && str[i] <= 'f')
        {
            v = (uint32_t)hex2str[(int)(str[i] - 'a')];
        }
        else
        {
            if (str[i] >= 'A' && str[i] <= 'F')
            {
                v = (uint32_t)hex2str[(int)(str[i] - 'A')];
            }
        }        
        int idx1 = j / 8; // array index
        addr->_limbs[idx1] |= (v << sht);
        sht = (sht + 4) % 32;
    }    
}

__host__ void print_bn(cgbn_mem_t<BITS> *a)
{
    int l = sizeof(a->_limbs) / sizeof(a->_limbs[0]);
    printf("BN (%d): ", l);
    for (int i = l - 1; i >= 0; i--)
    {
        printf("%08x", a->_limbs[i]);
    }
    printf("\n");
}

__host__ inline void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        printf("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

#define CHECKCUDAERR(ans)                          \
    {                                              \
        checkCudaError((ans), __FILE__, __LINE__); \
    }

__device__ env_t *cgbn_env[BATCH];

template <int N>
class ap_uint
{
private:
    env_t *bn_env = NULL;

    __device__ void get_env()
    {
        int tid = threadIdx.x;
        bn_env = cgbn_env[tid];
    }

public:
    env_t::cgbn_t a;

    __device__ ap_uint<N>(const cgbn_mem_t<BITS> *x)
    {
        get_env();
        cgbn_load(*bn_env, a, (cgbn_mem_t<BITS> *)x);
    }

    __device__ ap_uint<N>(const uint32_t x)
    {
        get_env();
        cgbn_set_ui32(*bn_env, a, x);
    }

    __device__ ap_uint<N>(const int x)
    {
        get_env();
        cgbn_set_ui32(*bn_env, a, (uint32_t)x);
    }

    __device__ ap_uint<N>()
    {
        get_env();
        cgbn_set_ui32(*bn_env, a, 0);
    }

    template <int M>
    __device__ ap_uint<N>(ap_uint<M> y)
    {
        get_env();
        cgbn_set(*bn_env, a, y.a);
    }

    __device__ ap_uint<N>(const char *str)
    {
        cgbn_mem_t<BITS> mem;
        str_to_cgbn_mem_t(&mem, str);
        cgbn_load(*bn_env, a, &mem);
    }

    __device__ inline ap_uint<N> &operator=(const ap_uint<N> &y)
    {
        cgbn_set(*bn_env, a, y.a);
        return *this;
    }

    template <int M>
    __device__ inline ap_uint<N> &operator=(const ap_uint<M> &y)
    {
        cgbn_set(*bn_env, a, y.a);
        return *this;
    }

    __device__ inline ap_uint<N> operator+(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        cgbn_add(*bn_env, z.a, a, y.a);
        return z;
    }

    __device__ inline ap_uint<N> operator++(int)
    {
        cgbn_add_ui32(*bn_env, a, a, 1);
        return *this;
    }

    __device__ inline ap_uint<N> operator+=(const ap_uint<N> &y)
    {
        *this = *this + y;
        return *this;
    }

    __device__ inline ap_uint<N> operator-(const ap_uint<N> &y)
    {
#ifdef TEST
/*
        if (*this < y)
        {
            printf("\nInvalid sub: a < b!\n");
            this->print();
            print(y);
        }
*/
#endif
        ap_uint<N> z;
        cgbn_sub(*bn_env, z.a, a, y.a);
        return z;
    }

    __device__ inline ap_uint<N> operator--(int)
    {
        cgbn_sub_ui32(*bn_env, a, a, 1);
        return *this;
    }

    __device__ inline ap_uint<N> operator-=(const ap_uint<N> &y)
    {
        *this = *this - y;
        return *this;
    }

    __device__ inline ap_uint<N> operator*(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        cgbn_mul(*bn_env, z.a, a, y.a);
        return z;
    }

    __device__ inline ap_uint<N> operator<<(const int &y)
    {
        ap_uint<N> z;
        cgbn_shift_left(*bn_env, z.a, a, (uint32_t)y);
        return z;
    }

    __device__ inline ap_uint<N> operator>>(const int &y)
    {
        ap_uint<N> z;
        cgbn_shift_right(*bn_env, z.a, a, (uint32_t)y);
        return z;
    }

    __device__ inline ap_uint<N> operator<<=(const int &y)
    {
        *this = *this << y;
        return *this;
    }

    __device__ inline ap_uint<N> operator>>=(const int &y)
    {
        *this = *this >> y;
        return *this;
    }

    __device__ inline int operator==(const ap_uint<N> &y)
    {
        return cgbn_equals(*bn_env, a, y.a);
    }

    __device__ inline int operator!=(const ap_uint<N> &y)
    {
        return !cgbn_equals(*bn_env, a, y.a);
    }

    __device__ inline int operator==(const int &y)
    {
        return cgbn_equals_ui32(*bn_env, a, (uint32_t)y);
    }

    __device__ inline int operator!=(const int &y)
    {
        return !cgbn_equals_ui32(*bn_env, a, (uint32_t)y);
    }

    __device__ inline int operator<(const ap_uint<N> &y)
    {
        return (cgbn_compare(*bn_env, a, y.a) < 0);
    }

    __device__ inline int operator>(const ap_uint<N> &y)
    {
        return (cgbn_compare(*bn_env, a, y.a) > 0);
    }

    __device__ inline int operator>(const int &y)
    {
        return (cgbn_compare_ui32(*bn_env, a, (uint32_t)y) > 0);
    }

    __device__ inline int operator<(const int &y)
    {
        return (cgbn_compare_ui32(*bn_env, a, (uint32_t)y) < 0);
    }

    __device__ inline int operator<=(const ap_uint<N> &y)
    {
        return (cgbn_compare(*bn_env, a, y.a) <= 0);
    }

    __device__ inline int operator>=(const ap_uint<N> &y)
    {
        return (cgbn_compare(*bn_env, a, y.a) >= 0);
    }

    __device__ inline uint32_t operator[](int pos)
    {
        ap_uint<1> y;
        cgbn_extract_bits(*bn_env, y.a, a, pos, 1);
        return (y == 1);
    }

    __device__ inline ap_uint<N> range(int end, int start)
    {
        ap_uint<N> y;
        cgbn_extract_bits(*bn_env, y.a, a, start, (end - start + 1));
        return y;
    }

    __device__ inline void set_range(int end, int start, ap_uint<N> y)
    {
        // env_t::cgbn_t tmp1, tmp2;
        // uint32_t size = (uint32_t)(end - start + 1);
        // cgbn_extract_bits(*bn_env, tmp1, a, 0, start);        
        // cgbn_shift_right(*bn_env, tmp2, a, end+1);
        // cgbn_shift_left(*bn_env, a, tmp2, end+1);
        // cgbn_bitwise_ior(*bn_env, a, tmp1, a);
        // cgbn_extract_bits(*bn_env, tmp1, y.a, 0, size);
        // cgbn_shift_left(*bn_env, tmp2, tmp1, start);
        // cgbn_bitwise_ior(*bn_env, a, a, tmp2);
        cgbn_insert_bits(*bn_env, a, a, start, (end - start + 1), y.a);
    }

    template <int M>
    __device__ inline ap_uint<M> range(int end, int start);
};

template <int N>
template <int M>
__device__ inline ap_uint<M> ap_uint<N>::range(int end, int start)
{
    ap_uint<M> y;
    cgbn_extract_bits(*bn_env, y.a, a, start, (end - start + 1));
    return y;
}
