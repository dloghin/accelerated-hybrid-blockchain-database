/*
 * Copyright 2022-2023 Rares Ifrim, Dumitrel Loghin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ap_uint_cpp.h"
#include <iostream>

ap_uint<256> a = ap_uint<256>("0");
ap_uint<256> p = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
ap_uint<256> n = ap_uint<256>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
ap_uint<256> Gx = ap_uint<256>("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
ap_uint<256> Gy = ap_uint<256>("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

template <int N>
ap_uint<N+1> productMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM) {
#pragma HLS inline off
    ap_uint<N + 1> tmp = 0;
PRODUCT_MOD:
    for (int i = N - 1; i >= 0; i--) {
        tmp <<= 1;
        if (tmp >= opM) {
            tmp -= opM;
        }
        if (opB[i] == 1) {
            tmp += opA;
            if (tmp >= opM) {
                tmp -= opM;
            }
        }
    }
    return tmp;
}

template <int N>
ap_uint<N> modularInv(ap_uint<N> opA, ap_uint<N> opM) {
    // calc r = opA^-1 * 2^k and k
    ap_uint<N> u = opA;
    ap_uint<N> v = opM;
    ap_uint<N> x1 = 1;
    ap_uint<N> x2 = 0;
    ap_uint<N+1> sum;
    
    INV_MOD:while (u != 1 && v != 1) {
        while(u[0] == 0){
            u = u >> 1;
            if(x1[0] == 0){
                x1 = x1 >> 1;
            }else{
                // x1 = (ap_uint<N+1>(x1 + opM)) >> 1;
                sum = opM;
                sum = sum + x1;
                x1 = sum >> 1;
            }
        }
        while(v[0] == 0){
            v = v >> 1;
            if(x2[0] == 0){
                x2 = x2 >> 1;
            }else{
                // x2 = (ap_uint<N+1>(x2 + opM)) >> 1;
                sum = opM;
                sum = sum + x2;
                x2 = sum >> 1;
            }
        }
        if(u >= v){
            u = u - v;
            if(x1 > x2){
                x1 = x1 - x2;
            }else{
                x1 = (x1 - x2) + opM;
            }
        }else{
            v = v - u;
            if(x2 > x1){
                x2 = x2 - x1;
            }else{
                x2 = (x2 - x1) + opM;
            }
        }
    }

    if(u == 1){
        if(x1 > opM){
            return x1 - opM;
        }else{
            return x1;
        }
    }else{
        if(x2 > opM){
            return x2 - opM;
        }else{
            return x2;
        }
    }
    
}

template <int N>
ap_uint<N> subMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM) {
    ap_uint<N + 1> opA1 = opA;
    ap_uint<N + 1> opB1 = opB;
    ap_uint<N + 1> opM1 = opM;
    ap_uint<N + 1> sum;
    if (opA >= opB) {
        sum = opA1 - opB1;
    } else {
        sum = opA1 + opM1;
        sum -= opB1;
    }
    return sum.range(N-1, 0);;
}

template <int N>
ap_uint<N> addMod(ap_uint<N> opA, ap_uint<N> opB, ap_uint<N> opM) {
    ap_uint<N + 1> opA1 = opA;
    ap_uint<N + 1> opB1 = opB;
    ap_uint<N + 1> sum = opA1 + opB1;
    if (sum >= opM) {
        sum -= opM;
    }
    return sum.range(N-1, 0);
}

 ap_uint<256> productMod_p4(ap_uint<256> a, ap_uint<256> b) {
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

        // std::cout << "s1: " << std::hex << s1 << std::endl;

        ap_uint<256> k11 = (s1 << 2) + (s1 << 1) + s1;
        ap_uint<256> k = (s1 << 32) + (k11 << 7) + (s1 << 6) + (s1 << 4) + s1;

        // std::cout << "k: " << std::hex << k << std::endl;
        // std::cout << "c0: " << std::hex << c0 << std::endl;

        ap_uint<256> tmp;                      
        tmp = addMod(k, c0, P);        
        tmp = addMod(tmp, w1, P);
        // std::cout << "w1: " << std::hex << w1 << std::endl;
        // std::cout << "tmp: " << std::hex << tmp << std::endl;
        tmp = addMod(tmp, w2, P);
        tmp = addMod(tmp, w3, P);
        tmp = addMod(tmp, w4, P);
        tmp = addMod(tmp, w5, P);
        tmp = addMod(tmp, w6, P);
        tmp = addMod(tmp, c1, P);
        
        // std::cout << "tmp: " << std::hex << tmp << std::endl;

        if (tmp >= P) {
            tmp -= P;
        }        

        return tmp;
}

ap_uint<256> productMod_p(ap_uint<256> a, ap_uint<256> b) { return productMod_p4(a, b); }

ap_uint<256> productMod_n(ap_uint<256> a, ap_uint<256> b) {
// XXX: a * b % n is only called a few times, no need to use specialized version
#pragma HLS inline
        return productMod<256>(a, b, n);
}

void fromJacobian(ap_uint<256> X, ap_uint<256> Y, ap_uint<256> Z, ap_uint<256>& x, ap_uint<256>& y) {
#pragma HLS inline
        if (Z == 0) {
            x = 0;
            y = 0;
        } else {
            ap_uint<256> ZInv = modularInv<256>(Z, p);
            ap_uint<256> ZInv_2 = productMod_p(ZInv, ZInv);
            ap_uint<256> ZInv_3 = productMod_p(ZInv_2, ZInv);
            x = productMod_p(X, ZInv_2);
            y = productMod_p(Y, ZInv_3);
        }
    }

void addJacobian(ap_uint<256> X1,
                     ap_uint<256> Y1,
                     ap_uint<256> Z1,
                     ap_uint<256> X2,
                     ap_uint<256> Y2,
                     ap_uint<256> Z2,
                     ap_uint<256>& X3,
                     ap_uint<256>& Y3,
                     ap_uint<256>& Z3) {
       
        ap_uint<256> I1 = productMod_p(Z1, Z1);
        ap_uint<256> I2 = productMod_p(Z2, Z2);
        ap_uint<256> J1 = productMod_p(I1, Z1);
        ap_uint<256> J2 = productMod_p(I2, Z2);
        ap_uint<256> U1 = productMod_p(X1, I2);
        ap_uint<256> U2 = productMod_p(X2, I1);
        ap_uint<256> H = subMod<256>(U1, U2, p);
        ap_uint<256> F = addMod<256>(H, H, p);
        F = productMod_p(F, F);        
        ap_uint<256> K1 = productMod_p(Y1, J2);
        ap_uint<256> K2 = productMod_p(Y2, J1);
        ap_uint<256> V = productMod_p(U1, F);
        ap_uint<256> G = productMod_p(F, H);        
        ap_uint<256> R = subMod<256>(K1, K2, p);        
        R = addMod<256>(R, R, p);        

        if (Z2 == 0) {
            X3 = X1;
            Y3 = Y1;
            Z3 = Z1;
        } else if (Z1 == 0) {
            X3 = X2;
            Y3 = Y2;
            Z3 = Z2;
        } else if (addMod<256>(K1, K2, p) == 0) {
            X3 = 1;
            Y3 = 1;
            Z3 = 0;
        } else {
            ap_uint<256> tmpX = productMod_p(R, R);
            ap_uint<256> tmp2V = addMod<256>(V, V, p);
            tmpX = addMod<256>(tmpX, G, p);
            X3 = subMod<256>(tmpX, tmp2V, p);

            ap_uint<256> tmp2 = subMod<256>(V, X3, p);
            tmp2 = productMod_p(tmp2, R);
            ap_uint<256> tmp4 = productMod_p(K1, G);
            tmp4 = addMod<256>(tmp4, tmp4, p);
            Y3 = subMod<256>(tmp2, tmp4, p);

            ap_uint<256> tmp5 = addMod<256>(Z1, Z2, p);
            tmp5 = productMod_p(tmp5, tmp5);
            ap_uint<256> tmp6 = addMod<256>(I1, I2, p);
            ap_uint<256> tmp7 = subMod<256>(tmp5, tmp6, p);
            Z3 = productMod_p(tmp7, H);
        }
    }

void doubleJacobian(
        ap_uint<256> X1, ap_uint<256> Y1, ap_uint<256> Z1, ap_uint<256>& X2, ap_uint<256>& Y2, ap_uint<256>& Z2) {
#pragma HLS inline
        ap_uint<256> N = productMod_p(Z1, Z1);
        ap_uint<256> E = productMod_p(Y1, Y1);
        ap_uint<256> B = productMod_p(X1, X1);
        ap_uint<256> L = productMod_p(E, E);

        ap_uint<256> tmp1 = addMod<256>(X1, E, p);
        tmp1 = productMod_p(tmp1, tmp1);
        ap_uint<256> tmp2 = addMod<256>(B, L, p);
        ap_uint<256> tmp3 = subMod<256>(tmp1, tmp2, p);
        ap_uint<256> S = addMod<256>(tmp3, tmp3, p);

        ap_uint<256> tmp4 = productMod_p(N, N);
        tmp4 = productMod_p(tmp4, a);
        ap_uint<256> tmp5 = addMod<256>(B, B, p);
        tmp5 = addMod<256>(tmp5, B, p);
        ap_uint<256> M = addMod<256>(tmp5, tmp4, p);

        ap_uint<256> tmp6 = addMod<256>(S, S, p);
        ap_uint<256> tmp7 = productMod_p(M, M);
        X2 = subMod<256>(tmp7, tmp6, p);

        ap_uint<256> tmp8 = subMod<256>(S, X2, p);
        tmp8 = productMod_p(tmp8, M);
        ap_uint<256> tmp9 = addMod<256>(L, L, p);
        tmp9 = addMod<256>(tmp9, tmp9, p);
        tmp9 = addMod<256>(tmp9, tmp9, p);
        Y2 = subMod<256>(tmp8, tmp9, p);

        ap_uint<256> tmp10 = addMod<256>(Y1, Z1, p);
        tmp10 = productMod_p(tmp10, tmp10);
        ap_uint<256> tmp11 = addMod<256>(E, N, p);
        Z2 = subMod<256>(tmp10, tmp11, p);
    }

void productJacobian(ap_uint<256> X1,
                         ap_uint<256> Y1,
                         ap_uint<256> Z1,
                         ap_uint<256> k,
                         ap_uint<256>& X2,
                         ap_uint<256>& Y2,
                         ap_uint<256>& Z2) {
#pragma HLS inline
        ap_uint<256> RX = 1;
        ap_uint<256> RY = 1;
        ap_uint<256> RZ = 0;
        
        ap_uint<256> tmpX, tmpY, tmpZ, tmpRX, tmpRY, tmpRZ;

        for (int i = 0; i < 1; i++) {
            std::cout << "RX: " << std::hex << RX << std::endl;
            std::cout << "RY: " << std::hex << RY << std::endl;
            std::cout << "RZ: " << std::hex << RZ << std::endl;
            std::cout << "X1: " << std::hex << X1 << std::endl;
            std::cout << "Y1: " << std::hex << Y1 << std::endl;
            std::cout << "Z1: " << std::hex << Z1 << std::endl; 
            addJacobian(RX, RY, RZ, X1, Y1, Z1, tmpRX, tmpRY, tmpRZ);
            std::cout << "tmpRX: " << std::hex << tmpRX << std::endl;
            std::cout << "tmpRY: " << std::hex << tmpRY << std::endl;
            std::cout << "tmpRZ: " << std::hex << tmpRZ << std::endl;             
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


ap_uint<8> verify(ap_uint<256> r, ap_uint<256> s, ap_uint<256> hash, ap_uint<256> Px, ap_uint<256> Py) {
#pragma HLS allocation function instances = productMod_p limit = 1
        if (r == 0 || r >= n || s == 0 || s >= n) {
            return ap_uint<8>(0);
        } else {
            ap_uint<256> z;
            z = hash;
            
            if (z >= n) {
                z -= n;
            }

            ap_uint<256> sInv = modularInv<256>(s, n);
             
            std::cout << "r: " << std::hex << r << std::endl;
            std::cout << "s: " << std::hex << s << std::endl;
            std::cout << "h: " << std::hex << hash << std::endl;
            std::cout << "Px: " << std::hex << Px << std::endl;
            std::cout << "Py: " << std::hex << Py << std::endl;
            std::cout << "sInv: " << std::hex << sInv << std::endl;            

            ap_uint<256> u1 = productMod_n(sInv, z);
            ap_uint<256> u2 = productMod_n(sInv, r);

            std::cout << "u1:" << std::hex << u1 << std::endl;
            std::cout << "u2:" << std::hex << u2 << std::endl;
            
            ap_uint<256> x, y;
          
            ap_uint<256> t1x, t1y, t1z, t2x, t2y, t2z;
            ap_uint<256> t3x, t3y, t3z;

            productJacobian(Gx, Gy, ap_uint<256>(1), u1, t1x, t1y, t1z);
            productJacobian(Px, Py, ap_uint<256>(1), u2, t2x, t2y, t2z);

            std::cout << "t1x:" << std::hex << t1x << std::endl;
            std::cout << "t1y:" << std::hex << t1y << std::endl;
            std::cout << "t1z:" << std::hex << t1z << std::endl;
            std::cout << "t2x:" << std::hex << t2x << std::endl;
            std::cout << "t2y:" << std::hex << t2y << std::endl;
            std::cout << "t2z:" << std::hex << t2z << std::endl;

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
            
            std::cout << "t3x:" << std::hex << t3x << std::endl;
            std::cout << "t3y:" << std::hex << t3y << std::endl;
            std::cout << "t3z:" << std::hex << t3z << std::endl;
            std::cout << "x:" << std::hex << x << std::endl;
            std::cout << "y:" << std::hex << y << std::endl;
            
            ap_uint<256> test_x, test_y, test_z;
    
    // addJacobian(ap_uint<256>("4D032A72E764B4C29E65A8199BC63F2B2AF8FCA8891017966590F55FF92BB1C2"),
    //             ap_uint<256>("F6184C705DF6BD98394726E3B99B1FD6703478F58A179EFAB735E98C36F5D6BA"),
    //             ap_uint<256>("50FF13D024D1B06517D7D564C1C715286CF211D61161C26F3891490E07AC8F19"),
    //             ap_uint<256>("22DB9C097EE3E65BBAB6D705CB645EC31FFF36320B3A166481866C8FECF43942"),
    //             ap_uint<256>("05A901DF67D455A7755D349A35E830A5B18743B4DA290AA0880E15B028D5D7DB"),
    //             ap_uint<256>("497D7E06C50F9DA5FF033A71779148F7912870CA9F0267D8095FF1C246B7A948"),
    //             test_x, test_y, test_z);
    
    // std::cout << "test_x:" << std::hex << test_x << std::endl;
    // std::cout << "test_y:" << std::hex << test_y << std::endl;
    // std::cout << "test_z:" << std::hex << test_z << std::endl;

            if (x == 0 && y == 0) {
                return ap_uint<8>(0);
            } else {
                if (r == x) {
                    return ap_uint<8>(1);
                } else {
                    return ap_uint<8>(0);
                }
            }
        }
}

#ifdef TEST

int main()
{
    // ap_uint<256> r = ap_uint<256>("9852734EFFF86C8D38A71D3D5F33E5F6804A5F98594DB4B2AB7A6A48651F7022");
    // ap_uint<256> s = ap_uint<256>("7576AED0B12EA1EAB03F8D4C39C59AC3C213E20CB9381A3144B37949C151E170");
    // ap_uint<256> h = ap_uint<256>("9230175B13981DA14D2F3334F321EB78FA0473133F6DA3DE896FEB22FB258936");
    // ap_uint<256> px = ap_uint<256>("C7B15D2BDF22B4351ECD30E7EEDF120124496A68B2280018CA817D1786F191B4");
    // ap_uint<256> py = ap_uint<256>("B5B5025F28B9DC46339F0C342DCEEC5F7F36E3EF77E0336657D5C215975368FE");

    ap_uint<256> r = ap_uint<256>("2c77652a760723a0ac5b32ba98e206e8eeee8cb285fc49139c4d2eb073ea25c4");
    ap_uint<256> s = ap_uint<256>("5ff5cb9a21b5ce7b403d461c88f0938d2404c56bbc88500906e8344f0c46f06c");
    ap_uint<256> h = ap_uint<256>("38f300a5a17105effbfd75c0a13e3ad2a4d65353eaf464df64fe4d3b2186aef8");
    // ap_uint<256> h = ap_uint<256>("38f300a5a17105effbfd75c0a13e3ad2a4d65353eaf464df64fe4d3b2186aef9");
    ap_uint<256> px = ap_uint<256>("c7b15d2bdf22b4351ecd30e7eedf120124496a68b2280018ca817d1786f191b4");
    ap_uint<256> py = ap_uint<256>("b5b5025f28b9dc46339f0c342dceec5f7f36e3ef77e0336657d5c215975368fe");

    r.print();
    s.print();
    h.print();
    px.print();
    py.print();

    // ap_uint<256> v = productMod_p4(s, r);
    // v.print();
    // v = productMod_n(s, r);
    // v.print();

    ap_uint<8> res = verify(r, s, h, px, py);
    res.print();
    // printf("Result %d\n", res[0]);
}
#endif
