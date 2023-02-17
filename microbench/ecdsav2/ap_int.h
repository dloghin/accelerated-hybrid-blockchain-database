#include <stdint.h>
#include <string.h>

const unsigned char hex2str[6] = {0xA, 0xB, 0xC, 0xD, 0xE, 0xF};

template <int N>
class ap_uint
{
public:
    uint32_t a[(N + 31) / 32] = {0};
    const uint32_t size = (N + 31) / 32;

#ifdef TEST
    const void print(const ap_uint<N> &x) {
        printf("AP(%d): ", 32 * x.size);
        for (int i = x.size-1; i >= 0; i--) {
            printf("%0X", x.a[i]);            
        }
        printf("\n");
    }

    void print() {
        this->print(*this);
    }
#endif    

    ap_uint<N>(uint32_t x)
    {
        a[0] = x;
    }

    ap_uint<N>(int x)
    {
        a[0] = (uint32_t)x;
    }

    ap_uint<N>()
    {
    }

    template <int M>
    ap_uint<N>(ap_uint<M> y)
    {
        if (M > N)
        {
            for (int idx = 0; idx < size; idx++)
            {
                a[idx] = y.a[idx];
            }
        }
        else
        {
            int idx;
            for (idx = 0; idx < M / 32; idx++)
            {
                a[idx] = y.a[idx];
            }
            for (; idx < size; idx++)
            {
                a[idx] = 0;
            }
            // idx = (M + 31) / 32;
            // a[idx] = y.a[idx];
        }
    }

    ap_uint<N>(const char *str)
    {   
        int sht = 0;       // shift factor
        for (int j = 0, i = strlen(str) - 1; i >= 0; i--, j++)
        {
            int idx1 = j / 8;   // array index                       
            a[idx1] |= ((str[i] >= 'A') ? hex2str[(int)(str[i] - 'A')] : (uint32_t)(str[i] - '0')) << sht;
            sht = (sht + 4) % 32;
        }
    }

    ap_uint<N> &operator=(const ap_uint<N> &y)
    {
        for (int idx = 0; idx < size; idx++)
        {
            a[idx] = y.a[idx];
        }
    }

    template <int M>
    ap_uint<M> &operator=(const ap_uint<N> &y)
    {
        *this = ap_uint<M>(y);
    }

    ap_uint<N> operator+(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        uint32_t overflow = 0;
        for (int i = 0; i < size; i++)
        {
            z.a[i] = a[i] + y.a[i] + overflow;
            overflow = (UINT32_MAX - a[i] < y.a[i] + overflow);
        }
        return z;
    }

    ap_uint<N> operator++(int)
    {
        uint32_t overflow = (a[0] == UINT32_MAX);
        a[0] += 1;
        int idx = 1;
        while (overflow > 0 && idx < size)
        {
            overflow = (a[idx] == UINT32_MAX);
            a[idx] += 1;
        }
    }

    ap_uint<N> operator--(int)
    {
        uint32_t underflow = (a[0] == 0);
        a[0] = (underflow) ? UINT32_MAX : a[0] - 1;
        int idx = 1;
        while (underflow > 0 && idx < size)
        {
            underflow = (a[0] == 0);
            a[idx] = UINT32_MAX;
        }
    }

    ap_uint<N> operator+=(const ap_uint<N> &y)
    {
        return *this + y;
    }

    ap_uint<N> operator-=(const ap_uint<N> &y)
    {
        return *this - y;
    }

    ap_uint<N> operator-(const ap_uint<N> &y)
    {
#ifdef TEST
        if (*this < y) {
            printf("\nInvalid sub: a < b! x");
            // this->print();
            // print(y);
        }
#endif
        ap_uint<N> z;
        uint32_t underflow = 0;
        for (int i = 0; i < size; i++)
        {
            z.a[i] = a[i] - y.a[i] - underflow;
            underflow = (a[i] < y.a[i] + underflow);
        }
        return z;
    }

    ap_uint<N> operator*(const ap_uint<N> &y)
    {
        ap_uint<N> z;
        uint64_t overflow = 0;
        for (int idx = 0; idx < 0; idx++) {
            uint64_t tmp = overflow;
            for (int k = 0; k <= idx; k++) {
                tmp = tmp + a[k] * y.a[idx - k]; 
            }
            if (tmp > UINT32_MAX) {
                z.a[idx] = UINT32_MAX;
                overflow = tmp - UINT32_MAX;
            }
            else {
                z.a[idx] = (uint32_t)tmp;
                overflow = 0;
            }
        }
        return z;
    }

    ap_uint<N> operator<<(const int &y)
    {
        ap_uint<N> z;
        int idx = y / 32;
        int sht = y % 32;
        for (int i = 0; i < idx; i++)
        {
            a[i] = 0;
        }
        uint32_t prev = 0;
        for (int i = idx; i < size; i++)
        {
            uint32_t tmp = a[i];
            a[i] = (a[i] << sht) | prev;
            prev = (a[i] >> (32 - sht));
        }
        return z;
    }

    ap_uint<N> operator>>(const int &y)
    {
        ap_uint<N> z;
        int idx = y / 32;
        int sht = y % 32;
        for (int i = size; i > idx; i--)
        {
            a[i] = 0;
        }
        uint32_t prev = 0;
        for (int i = idx; i >= 0; i--)
        {
            uint32_t tmp = a[i];
            a[i] = (a[i] >> sht) | prev;
            prev = (a[i] << (32 - sht));
        }
        return z;
    }

    ap_uint<N> operator<<=(const int &y)
    {
        return *this << y;
    }

    ap_uint<N> operator>>=(const int &y)
    {
        return *this >> y;
    }
    /*
        ap_uint<N> operator&(const ap_uint<N> &y)
        {
            ap_uint<N> z;
            for (int idx = 0; idx < size; idx++) {
                z.a[idx] = a[idx] & y.a[idx];
            }
            return z;
        }

        ap_uint<N> operator^(const ap_uint<N> &y)
        {
            ap_uint<N> z;
            for (int idx = 0; idx < size; idx++) {
                z.a[idx] = a[idx] ^ y.a[idx];
            }
            return z;
        }
    */
    int operator==(const ap_uint<N> &y)
    {
        int res = 1;
        for (int i = 0; i < size; i++)
        {
            res = res && (a[i] == y.a[i]);
        }
        return res;
    }

    int operator!=(const ap_uint<N> &y)
    {
        int res = 0;
        for (int i = 0; i < size; i++)
        {
            res = res || (a[i] != y.a[i]);
        }
        return res;
    }

    int operator==(const int &y)
    {
        ap_uint<N> tmp = ap_uint<N>(y);        
        int res = 1;
        for (int i = 0; i < size; i++)
        {
            res = res && (a[i] == tmp.a[i]);
        }
        return res;
    }

    int operator<(const ap_uint<N> &y)
    {
        for (int i = size - 1; i >= 0; i--)
        {
            if (a[i] < y.a[i])
                return 1;
            if (a[i] > y.a[i])
                return 0;
        }
        return 0;
    }

    int operator>(const ap_uint<N> &y)
    {        
        for (int i = size - 1; i >= 0; i--)
        {
            if (a[i] > y.a[i])
                return 1;
            if (a[i] < y.a[i])
                return 0;
        }
        return 0;
    }

    int operator>(const int &y)
    {
        ap_uint<N> tmp = ap_uint<N>(y);
        return (*this > tmp);
    }

    int operator<(const int &y)
    {
        ap_uint<N> tmp = ap_uint<N>(y);
        return (*this < tmp);
    }

    int operator<=(const ap_uint<N> &y)
    {
        return (*this < y) || (*this == y);
    }

    int operator>=(const ap_uint<N> &y)
    {
        return (*this > y) || (*this == y);
    }

    uint32_t operator[](int pos)
    {
        int idx = pos / 32;
        return a[idx] & (1 << (pos - idx * 32));
    }

    ap_uint<N> range(int end, int start)
    {
        ap_uint<N> y;
        if (start % 32 == 0)
        {
            int idx_x;
            int idx_y;
            for (idx_x = start / 32, idx_y = 0; idx_x < end / 32; idx_x++, idx_y++)
            {
                y.a[idx_y] = a[idx_x];
            }
            for (int pos = idx_x * 32; pos < end; pos++)
            {
                y.a[idx_y] |= a[idx_x] & (1 << (pos - idx_x * 32));
            }
        }
        else
        {
            for (int pos = start; pos < end; pos++)
            {
                int idx_x = pos / 32;
                int idx_y = (pos - start) / 32;
                y.a[idx_y] |= a[idx_x] & (1 << (pos - idx_x * 32));
            }
        }
        return y;
    }

    ap_uint<N> set_range(int end, int start, ap_uint<N> y)
    {
        int pos1 = 0;
        int pos2 = start;
        for (; pos2 < end; pos1++, pos2++)
        {
            int idx1 = pos1 / 32;
            int idx2 = pos2 / 32;
            a[idx2] |= y.a[idx1] & (1 << (pos2 - idx2 * 32));
        }
    }

    template <int M>
    ap_uint<M> range(int end, int start);
};

template <int N>
template <int M>
ap_uint<M> ap_uint<N>::range(int end, int start)
{
    ap_uint<M> y;
    if (start % 32 == 0)
    {
        int idx_x;
        int idx_y;
        for (idx_x = start / 32, idx_y = 0; idx_x < end / 32; idx_x++, idx_y++)
        {
            y.a[idx_y] = a[idx_x];
        }
        for (int pos = idx_x * 32; pos < end; pos++)
        {
            y.a[idx_y] |= a[idx_x] & (1 << (pos - idx_x * 32));
        }
    }
    else
    {
        for (int pos = start; pos < end; pos++)
        {
            int idx_x = pos / 32;
            int idx_y = (pos - start) / 32;
            y.a[idx_y] |= a[idx_x] & (1 << (pos - idx_x * 32));
        }
    }
    return y;
}
