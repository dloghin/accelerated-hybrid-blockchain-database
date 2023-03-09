#include <stdint.h>
#include <string.h>
#include <iostream>

#include <gmp.h>

const unsigned char hex2str[6] = {0xA, 0xB, 0xC, 0xD, 0xE, 0xF};

template <int N>
class ap_uint
{
public:
    mpz_t a;

#ifdef TEST
    const void print(const ap_uint<N> &x)
    {
        gmp_printf("%#Zx\n", x.a);
    }

    const void print()
    {
        this->print(*this);
    }

    friend std::ostream& operator<<(std::ostream& os, const ap_uint<N> &x)
    {
        char buff[128];
        gmp_sprintf(buff, "%#Zx", x.a);
        os << buff;
        return os;
    }
#endif

    ap_uint<N>(uint32_t x)
    {
        mpz_init(a);
        mpz_set(a, x);
    }

    ap_uint<N>(uint64_t x)
    {
        mpz_init(a);
        mpz_set_ui(a, x);
    }

    ap_uint<N>(int x)
    {
        mpz_init(a);
        mpz_set_ui(a, (uint32_t)x);
    }

    ap_uint<N>()
    {
        mpz_init(a);
    }

    template <int M>
    ap_uint<N>(ap_uint<M> y)
    {
        mpz_init(a);
        mpz_set(a, y.a);
    }

    ap_uint<N>(const char *str)
    {       
        mpz_init(a);
        mpz_set_str(a, str, 16);
    }

    ap_uint<N> &operator=(const ap_uint<N> &y)
    {        
        mpz_t tmp;
        mpz_init(tmp);
        mpz_set(tmp, y.a);
        mpz_init(a);
        mpz_set(a, tmp);
        return *this;
    }

    template <int M>
    ap_uint<N> &operator=(const ap_uint<M> &y)
    {
        mpz_t tmp;
        mpz_init(tmp);
        mpz_set(tmp, y.a);
        mpz_init(a);
        mpz_set(a, tmp);
        return *this;
    }

    ap_uint<N> operator+(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        mpz_add(z.a, a, y.a);
        return z;
    }

    ap_uint<N> operator++(int)
    {
        mpz_add(a, a, 1);
        return *this;
    }

    ap_uint<N> operator+=(const ap_uint<N> &y)
    {
        *this = *this + y;
        return *this;
    }

    ap_uint<N> operator-(const ap_uint<N> &y)
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
        mpz_sub(z.a, a, y.a);
        return z;
    }

    ap_uint<N> operator--(int)
    {
        mpz_sub_ui(a, a, 1);
        return *this;
    }

    ap_uint<N> operator-=(const ap_uint<N> &y)
    {
        *this = *this - y;
        return *this;
    }

    ap_uint<N> operator*(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        mpz_mul(z.a, a, y.a);
        return z;
    }

    ap_uint<N> operator<<(const int &y)
    {
        ap_uint<N> z;
        mpz_mul_2exp(z.a, a, y);
        mpz_tdiv_r_2exp(z.a, z.a, N);
        return z;
    }

    ap_uint<N> operator>>(const int &y)
    {
        ap_uint<N> z;
        mpz_fdiv_q_2exp(z.a, a, y); 
        return z;
    }

    ap_uint<N> operator<<=(const int &y)
    {
        *this = *this << y;
        return *this;
    }

    ap_uint<N> operator>>=(const int &y)
    {
        *this = *this >> y;
        return *this;
    }

    int operator==(const ap_uint<N> &y)
    {
        return (mpz_cmp(a, y.a) == 0);
    }

    int operator!=(const ap_uint<N> &y)
    {
        return (mpz_cmp(a, y.a) != 0);
    }

    int operator==(const int &y)
    {
        return (mpz_cmp_ui(a, y) == 0);
    }

    int operator<(const ap_uint<N> &y)
    {        
        return (mpz_cmp(a, y.a) < 0);
    }

    int operator>(const ap_uint<N> &y)
    {
        return (mpz_cmp(a, y.a) > 0);
    }

    int operator>(const int &y)
    {
        return (mpz_cmp_ui(a, y) > 0);
    }

    int operator<(const int &y)
    {
        return (mpz_cmp_ui(a, y) < 0);
    }

    int operator<=(const ap_uint<N> &y)
    {
        return (mpz_cmp(a, y.a) <= 0);
    }

    int operator>=(const ap_uint<N> &y)
    {
        return (mpz_cmp(a, y.a) >= 0);
    }

    uint32_t operator[](int pos)
    {
        return mpz_tstbit(a, pos);
    }

    ap_uint<N> range(int end, int start)
    {
        int i, k;
        ap_uint<N> y;
        mpz_init(y.a);
        for (i = start, k = 0; i <= end; i++, k++)
        {
            if (mpz_tstbit(a, i))
                mpz_setbit(y.a, k);
        }
        return y;
    }

    void set_range(int end, int start, ap_uint<N> y)
    {
        int i, k;
        for (i = start, k = 0; i <= end; i++, k++)
        {
            if (mpz_tstbit(y.a, k))
                mpz_setbit(a, i);
        }
    }

    template <int M>
    ap_uint<M> range(int end, int start);
};

template <int N>
template <int M>
ap_uint<M> ap_uint<N>::range(int end, int start)
{
    int i, k;
    ap_uint<M> y;
    mpz_init(y.a);
    for (i = start, k = 0; i <= end; i++, k++)
    {
        if (mpz_tstbit(a, i))
            mpz_setbit(y.a, k);
    }
    return y;
}
