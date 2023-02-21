#ifdef TEST
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#endif

#include "ap_int.h"

ap_uint<256> a = ap_uint<256>("0");
ap_uint<256> p = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
ap_uint<256> n = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
ap_uint<256> Gx = ap_uint<256>("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
ap_uint<256> Gy = ap_uint<256>("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

template <int N>
ap_uint<N> productMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
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
ap_uint<N> monProduct(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
    ap_uint<N + 2> s = 0;
    ap_uint<1> a0 = opA[0];
    for (int i = 0; i < N; i++)
    {
        ap_uint<1> qa = opB[i];
        ap_uint<1> qm = ap_uint<1>(s[0] ^ (opB[i] & a0[0]));
        ap_uint<N> addA = qa == ap_uint<1>(1) ? opA : ap_uint<N>(0);
        ap_uint<N> addM = qm == ap_uint<1>(1) ? opM : ap_uint<N>(0);
        s += (addA + addM);
        s >>= 1;
    }
    if (s > opM)
    {
        s -= opM;
    }
    return s;
}

template <int N>
ap_uint<N> modularInv_v1(ap_uint<N> opA, ap_uint<N> opM)
{
    // calc r = opA^-1 * 2^k and k
    ap_uint<N> u = opM;
    ap_uint<N> v = opA;
    ap_uint<N> s = 1;
    ap_uint<N + 1> r = 0;
    ap_uint<32> k = 0;

    while (v > 0)
    {        
        if (u[0] == 0)
        {            
            u >>= 1;
            s <<= 1;
        }
        else if (v[0] == 0)
        {            
            v = v >> 1;
            r <<= 1;
        }
        else if (u > v)
        {            
            u -= v;
            u >>= 1;
            r += s;
            s <<= 1;
        }
        else
        {        
            v -= u;
            v >>= 1;
            s += r;
            r <<= 1;
        }
        k++;
    }

    ap_uint<N + 1> opM2 = ap_uint<N + 1>(opM);
    if (r >= opM2)
    {
        r = r - opM2;
    }
    r = opM2 - r;

    if (k < N)
    {
        k = ap_uint<32>(256);
    }
    else
    {
        k = k - ap_uint<32>(N);
    }

    // here, since k is ap_uint<32>, then k == k[0]
    for (int i = 0; i < k[0]; i++)
    {
        if (r[0] == 1)
        {
            r += opM;
        }
        r >>= 1;
    }

    ap_uint<N + 1> tmp1 = r.range(N - 1, 0);
    ap_uint<N> tmp2 = ap_uint<N>(tmp1);
    ap_uint<N> res = monProduct<N>(tmp2, 1, opM);

    return res;
}

template <int N>
ap_uint<N> modularInv(ap_uint<N> opA, ap_uint<N> opM)
{
    // calc r = opA^-1 * 2^k and k
    ap_uint<N> u = opA;
    ap_uint<N> v = opM;
    ap_uint<N> x1 = ap_uint<N>(1);
    ap_uint<N> x2 = ap_uint<N>(0);
    const ap_uint<N> one = ap_uint<N>(1);
    ap_uint<N + 1> opM2 = ap_uint<N + 1>(opM);

    while (u != one && v != one)
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
                x1 = (opM + x1) >> 1;
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
                x2 = (opM + x2) >> 1;
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
                while (x1 < x2)
                    x1 = x1 + opM;
                x1 = x1 - x2;
                // x1 = (x1 + opM) - x2;
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
                while (x2 < x1)
                    x2 = x2 + opM;
                x2 = x2 - x1;
                // x2 = (x2 + opM) - x1;
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
ap_uint<N> subMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
#ifdef TEST
    if (opA > opM)
    {
        printf("In subMod: opA > opM!\n");
    }
    if (opB > opM)
    {
        printf("In subMod: opB > opM!\n");
    }
#endif
    ap_uint<N + 1> sum = ap_uint<N + 1>(opA);
    if (opA >= opB)
    {
        sum = sum - opB;
    }
    else
    {
        sum = sum + opM;
        sum = sum - opB;
    }
    return sum;
}

template <int N>
ap_uint<N> addMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM)
{
    ap_uint<N + 1> sum = opA + opB;
    if (sum >= opM)
    {
        sum -= opM;
    }
    return sum;
}

ap_uint<256> productMod_p4(ap_uint<256> a, ap_uint<256> b)
{
    ap_uint<128> aH = a.range<128>(255, 128);
    ap_uint<128> aL = a.range<128>(127, 0);
    ap_uint<128> bH = b.range<128>(255, 128);
    ap_uint<128> bL = b.range<128>(127, 0);

    ap_uint<256> aLbH = aL * bH;
    ap_uint<256> aHbL = aH * bL;
    ap_uint<512> aHbH = aH * bH;
    ap_uint<256> aLbL = aL * bL;
    ap_uint<512> mid = ap_uint<512>(aLbH) + ap_uint<512>(aHbL);

    ap_uint<512> mul = (aHbH << 256) + (mid << 128) + aLbL;
    ap_uint<256> P = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    ap_uint<256> c0 = mul.range<256>(255, 0);
    ap_uint<256> c1 = mul.range<256>(511, 256);
    ap_uint<256> w1 = ap_uint<256>(0);
    ap_uint<256> w2 = ap_uint<256>(0);
    ap_uint<256> w3 = ap_uint<256>(0);
    ap_uint<256> w4 = ap_uint<256>(0);
    ap_uint<256> w5 = ap_uint<256>(0);
    ap_uint<256> w6 = ap_uint<256>(0);

    /*
            w1.range(255, 32) = c1.range(223, 0);
            w2.range(255, 9) = c1.range(246, 0);
            w3.range(255, 8) = c1.range(247, 0);
            w4.range(255, 7) = c1.range(248, 0);
            w5.range(255, 6) = c1.range(249, 0);
            w6.range(255, 4) = c1.range(251, 0);
    */

    w1.set_range(255, 32, c1.range(223, 0));
    w2.set_range(255, 9, c1.range(246, 0));
    w3.set_range(255, 8, c1.range(247, 0));
    w4.set_range(255, 7, c1.range(248, 0));
    w5.set_range(255, 6, c1.range(249, 0));
    w6.set_range(255, 4, c1.range(251, 0));

    ap_uint<256> s1 = c1.range<256>(255, 252) + c1.range<256>(255, 250) + c1.range<256>(255, 249) + c1.range<256>(255, 248) +
                      c1.range<256>(255, 247) + c1.range<256>(255, 224);
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

ap_uint<256> productMod_p(ap_uint<256> a, ap_uint<256> b) { return productMod_p4(a, b); }

ap_uint<256> productMod_n(ap_uint<256> a, ap_uint<256> b)
{
    // XXX: a * b % n is only called a few times, no need to use specialized version
    return productMod<256>(a, b, n);
}

void add(ap_uint<256> Px, ap_uint<256> Py, ap_uint<256> Qx, ap_uint<256> Qy, ap_uint<256> &Rx, ap_uint<256> &Ry)
{
    if (Qx == 0 && Qy == 0)
    { // Q is zero
        Rx = Px;
        Ry = Py;
    }
    else if (Px == Qx && (Py + Qy) == p)
    {
        Rx = 0;
        Ry = 0;
    }
    else
    {
        ap_uint<256> lamda, lamda_d;
        if (Px == Qx && Py == Qy)
        {
            lamda = productMod_p(Px, Px);
            lamda = productMod_p(lamda, ap_uint<256>(3));
            lamda = addMod<256>(lamda, a, p);
            lamda_d = productMod_p(Py, ap_uint<256>(2));
        }
        else
        {
            lamda = subMod<256>(Qy, Py, p);
            lamda_d = subMod<256>(Qx, Px, p);
        }
        lamda_d = modularInv<256>(lamda_d, p);
        lamda = productMod_p(lamda, lamda_d);
        ap_uint<256> lamda_sqr = productMod_p(lamda, lamda);
        ap_uint<256> resX, resY;
        resX = subMod<256>(lamda_sqr, Px, p);
        resX = subMod<256>(resX, Qx, p);

        resY = subMod<256>(Px, resX, p);
        resY = productMod_p(lamda, resY);
        resY = subMod<256>(resY, Py, p);

        Rx = resX;
        Ry = resY;
    }
}

void scalarProduct(ap_uint<256> s, ap_uint<256> Px, ap_uint<256> Py, ap_uint<256> &Rx, ap_uint<256> &Ry)
{

    ap_uint<256> tempRx = ap_uint<256>(0);
    ap_uint<256> tempRy = ap_uint<256>(0);
    ap_uint<256> tempPx;
    ap_uint<256> tempPy;

    for (int i = 0; i < 256; i++)
    {
        if (s[i])
        {
            add(Px, Py, tempRx, tempRy, Rx, Ry);
            tempRx = Rx;
            tempRy = Ry;
        }

        add(Px, Py, Px, Py, tempPx, tempPy);
        Px = tempPx;
        Py = tempPy;
    }
}

ap_uint<8> verify(ap_uint<256> r, ap_uint<256> s, ap_uint<256> hash, ap_uint<256> Px, ap_uint<256> Py)
{
    if (r == 0 || r >= n || s == 0 || s >= n)
    {
#ifdef TEST
        printf("No verification: invalid signature.\n");
#endif
        return ap_uint<8>(0);
    }
    else
    {
        ap_uint<256> z;
        z = hash;

        if (z >= n)
        {
            z -= n;
        }

printf("here\n");

        ap_uint<256> sInv = modularInv<256>(s, n);

        ap_uint<256> u1 = productMod_n(sInv, z);
        ap_uint<256> u2 = productMod_n(sInv, r);

        ap_uint<256> t1x, t1y, t2x, t2y;
        scalarProduct(u1, Gx, Gy, t1x, t1y);
        scalarProduct(u2, Px, Py, t2x, t2y);

        ap_uint<256> x, y;
        add(t1x, t1y, t2x, t2y, x, y);

        r.print();
        x.print();

        if (x == 0 && y == 0)
        {
            printf("x 0 y 0\n");
            return ap_uint<8>(0);
        }
        else
        {
            if (r == x)
            {
                printf("r == x\n");
                return ap_uint<8>(1);
            }
            else
            {
                printf("r != x\n");
                return ap_uint<8>(0);
            }
        }
    }
}

#ifdef TEST

#define TEST_LOOPS 1000

void test_add()
{
    srand(time(NULL));
    uint64_t ua, ub;

    for (int i = 0; i < TEST_LOOPS; i++)
    {
        ua = (uint64_t)UINT32_MAX + 10 * rand();
        ub = (uint64_t)UINT32_MAX + 7 * rand();
        ap_uint<64> aa = ap_uint<64>(ua);
        ap_uint<64> ab = ap_uint<64>(ub);
        aa = aa + ab;
        ua = ua + ub;
        ap_uint<64> ac = ap_uint<64>(ua);
        if (ac != aa)
        {
            printf("Invalid ADD result.\n");
            return;
        }
    }
    printf("All ADDs are valid.\n");
}

void test_sub()
{
    srand(time(NULL));
    uint64_t ua, ub;

    for (int i = 0; i < TEST_LOOPS; i++)
    {
        ua = UINT32_MAX + 10 * rand();
        ub = ua - 7;
        ap_uint<64> aa = ap_uint<64>(ua);
        ap_uint<64> ab = ap_uint<64>(ub);
        aa = aa - ab;
        ua = ua - ub;
        ap_uint<64> ac = ap_uint<64>(ua);
        if (ac != aa)
        {
            printf("Invalid SUB result.\n");
            return;
        }
    }
    printf("All SUBs are valid.\n");
}

void test_mul()
{
    srand(time(NULL));
    uint64_t ua, ub, uc;

    for (int i = 0; i < TEST_LOOPS; i++)
    {
        ua = UINT32_MAX + 10 * rand();
        ub = UINT32_MAX + 7 * rand();
        ap_uint<64> aa = ap_uint<64>(ua);
        ap_uint<64> ab = ap_uint<64>(ub);
        aa = aa * ab;
        uc = ua * ub;
        ap_uint<64> ac = ap_uint<64>(uc);
        if (ac != aa)
        {
            printf("Invalid MUL result: %lx * %lx = %lx\n", ua, ub, uc);
            aa.print();
            return;
        }
    }
    printf("All MULs are valid.\n");
}

void test_shl()
{
    srand(time(NULL));
    uint64_t ua, ub;

    for (int i = 0; i < TEST_LOOPS; i++)
    {
        ua = UINT32_MAX + 67345 * rand();
        ap_uint<64> aa = ap_uint<64>(ua);
        int r = rand() % 64;
        ap_uint<64> ab = aa << r;
        ub = ua << r;
        ap_uint<64> ac = ap_uint<64>(ub);
        if (ac != ab)
        {
            printf("Invalid Shift Left result: %lx << %d = %lx\n", ua, r, ub);
            aa.print();
            ab.print();
            return;
        }
    }
    printf("All Shift Left are valid.\n");
}

void test_shr()
{
    srand(time(NULL));
    uint64_t ua, ub;

    for (int i = 0; i < TEST_LOOPS; i++)
    {
        ua = UINT32_MAX + 67345 * rand();
        ap_uint<64> aa = ap_uint<64>(ua);
        int r = rand() % 64;
        ap_uint<64> ab = aa >> r;
        ub = ua >> r;
        ap_uint<64> ac = ap_uint<64>(ub);
        if (ac != ab)
        {
            printf("Invalid Shift Right result: %lx >> %d = %lx\n", ua, r, ub);
            aa.print();
            ab.print();
            return;
        }
    }
    printf("All Shift Right are valid.\n");
}

void test_gtlt()
{
    srand(time(NULL));
    uint64_t ua, ub;

    for (int i = 0; i < TEST_LOOPS; i++)
    { 
        ua = 3 * (uint64_t)UINT32_MAX; // + (uint64_t)(90 * rand());
        ub = 2 * (uint64_t)UINT32_MAX; // + (uint64_t)(70 * rand());
        ap_uint<64> aa = ap_uint<64>(ua);
        ap_uint<64> ab = ap_uint<64>(ub);

        if ((ua >= ub) && (aa < ab || aa <= ab)) {
            printf("Invalid < or <=\n");
            aa.print();
            ab.print();
            return;
        }

        if ((ua >= ub) && (ab > aa || ab >= aa)) {
            printf("Invalid > or >=\n%lx %lx\n", ua, ub);
            aa.print();
            ab.print();
            return;
        }
    }
    printf("All <,>,<=,>= are valid.\n");
}

int main()
{
    // ap_uint<256> r = ap_uint<256>("9852734EFFF86C8D38A71D3D5F33E5F6804A5F98594DB4B2AB7A6A48651F7022");
    // ap_uint<256> s = ap_uint<256>("7576AED0B12EA1EAB03F8D4C39C59AC3C213E20CB9381A3144B37949C151E170");
    // ap_uint<256> h = ap_uint<256>("9230175B13981DA14D2F3334F321EB78FA0473133F6DA3DE896FEB22FB258936");
    // ap_uint<256> px = ap_uint<256>("C7B15D2BDF22B4351ECD30E7EEDF120124496A68B2280018CA817D1786F191B4");
    // ap_uint<256> py = ap_uint<256>("B5B5025F28B9DC46339F0C342DCEEC5F7F36E3EF77E0336657D5C215975368FE");

    ap_uint<256> r = ap_uint<256>("7E5790FC5C01FA6622DDEAF38A0CF83D9A43EA3F9C18EDECD6F6F9424B619DBB");
    ap_uint<256> s = ap_uint<256>("4232D32296B01568889322A7C2434D573BE61BC91EA81FED998CC6CEDE3109A4");
    ap_uint<256> h = ap_uint<256>("BB0E1853B125071FDA800F1581AE63B351CBD874D55636AA846C0D58B82592AC");
    ap_uint<256> px = ap_uint<256>("20E4DB638AE55585178FA25F764986B4E9D812EE74CE876832B9C71E13DCCF68");
    ap_uint<256> py = ap_uint<256>("8A6DA3C4643262337B65B8173E147DB1D468C088569BBD42B45D65E376116329");

    r.print();
    s.print();
    h.print();
    px.print();
    py.print();
    n.print();

    ap_uint<256> c1 = ap_uint<256>("2");
    ap_uint<256> c2 = ap_uint<256>("4");

    c1.print();
    c2.print();
    // c1 = c1 >> 1;
    // c2 = c2 << 1;
    // c1.print();
    // c2.print();
    ap_uint<256> c3 = c2 * c1;
    c3.print();

    ap_uint<8> res = verify(r, s, h, px, py);
    res.print();
    // printf("Result %d\n", res[0]);

    test_add();
    test_sub();
    test_mul();
    test_shl();
    test_shr();
    test_gtlt();
}
#endif
