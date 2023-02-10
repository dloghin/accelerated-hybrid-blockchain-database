/*
 * Copyright 2022 Dumitrel Loghin
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

package main

// #cgo CFLAGS: -I/home/dumi/git/RapidEC/src
// #cgo LDFLAGS: -L/home/dumi/git/RapidEC/src -lverify -lgomp -lgmp
// #include "hbdb_interface.h"
import "C"

import (
	"fmt"
	"hash"
	"encoding/hex"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"
	"crypto/rand"
	"github.com/tjfoc/gmsm/sm2"

	"go.uber.org/atomic"
	"golang.org/x/crypto/sha3"
	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"	
)

type Pair struct {
	a, b []byte
}

var (
	dataRun       = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()	
	saveResults   = kingpin.Flag("save", "Save digests to output-cpu.txt").Bool()
)

type KeccakState interface {
	hash.Hash
	Read([]byte) (int, error)
}

func myKeccak256(data ...[]byte) []byte {
	b := make([]byte, 32)
	d := sha3.New256().(KeccakState)
	for _, b := range data {
		d.Write(b)
	}
	d.Read(b)
	return b
}

func mySm2Sign(priv *sm2.PrivateKey, msg []byte) ([]byte, error) {
	return priv.Sign(rand.Reader, msg, nil)	
}

func mySm2SignRS(priv *sm2.PrivateKey, msg []byte) ([]byte, error) {
	sig, _ := priv.Sign(rand.Reader, msg, nil)
	ret := make([]byte, 64)
	lr := int(sig[3])
	copy(ret[:32], sig[4:4+lr])	
	copy(ret[32:], sig[6+lr:])
	return ret, nil
}

func mySm2Verify(pub *sm2.PublicKey, msg, sig []byte) bool {
	return pub.Verify(msg, sig)
}

func main() {
	kingpin.Parse()

	// generate key
	pvk, err := sm2.GenerateKey(rand.Reader)
	if err != nil {
		panic(err)
	}
	pub := &pvk.PublicKey

	pubkey := make([]byte, 64)
	copy(pubkey[:32], pub.X.Bytes())
	copy(pubkey[32:], pub.Y.Bytes())

	fmt.Println("Load data ...")
	runFile, err := os.Open(*dataRun)
	if err != nil {
		panic(err)
	}
	defer runFile.Close()
	runBuf := make(chan Pair, 100000)
	reqNum := atomic.NewInt64(0)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(runBuf)
		if err := benchmark.LineByLine(runFile, func(line string) error {
			reqNum.Add(1)
			operands := strings.SplitN(line, " ", 5)
			l := len(operands[2])
			// if l%8 != 0 {
			//	l = 8 * (l/8 + 1)
			// }
			// copy data
			buf := make([]byte, l)
			copy(buf, operands[2])
			msg := myKeccak256(buf)
			sig, err := mySm2SignRS(pvk, msg)
			if err != nil {
				fmt.Println(err)
			}
			runBuf <- Pair{buf, sig}
			return nil
		}); err != nil {
			panic(err)
		}
	}()
	time.Sleep(5 * time.Second)

	var outFile *os.File
	if *saveResults {
		outFile, err = os.Create("output-gpu.txt")
		if err != nil {
			fmt.Printf("Error creating output file: %v\n", err)
		}
	}

	// Run on GPU
	fmt.Println("Run on GPU ...")
	num := 32
	cnum := C.int(num)
	C.init_gpu(cnum)
	data := make([]byte, num*256)
	sizes := make([]C.int, num)
	signatures := make([]byte, num*64)
	keys := make([]byte, num*64)
	idx := 0
	offset := 0
	// run in batches
	start := time.Now()
	for pair := range runBuf {
		l := len(pair.a)
		// copy data
		sizes[idx] = C.int(l)
		copy(data[offset:], pair.a)
		copy(signatures[idx*64:(idx+1)*64], pair.b[:64])
		copy(keys[idx*64:(idx+1)*64], pubkey)
		offset += l
		// fmt.Printf("Data size: %v\n", sizes[idx])
		// fmt.Printf("Data: %v\n", string(data[idx]))
		idx++
		if idx == num {
			// call kernel
			dptr := (*C.uchar)(unsafe.Pointer(&data[0]))
			sptr := (*C.int)(unsafe.Pointer(&sizes[0]))
			kptr := (*C.uchar)(unsafe.Pointer(&keys[0]))
			gptr := (*C.uchar)(unsafe.Pointer(&signatures[0]))
			// fmt.Printf("Size %v Offset %d\n", size, offset)
			C.run_kernel(dptr, sptr, kptr, gptr, cnum)
			if *saveResults {
				// var rptr *C.int
				res := C.GoBytes(unsafe.Pointer(C.get_results(cnum)), cnum)
				hashes := C.GoBytes(unsafe.Pointer(C.get_hashes(cnum)), 3200)
				for i := 0; i < num; i++ {
					if res[i] == 1 {						
						outFile.WriteString(hex.EncodeToString(hashes[32*i:32*i+32]) + " 1\n")
					} else {
						outFile.WriteString(hex.EncodeToString(hashes[32*i:32*i+32]) + " 0\n")
					}
				}
			}

			// reset counters
			idx = 0
			offset = 0
		}
	}
	delta := time.Since(start).Seconds()
	fmt.Printf("Throughput on GPU to handle %v requests: %v req/s\n",
		reqNum, int64(float64(reqNum.Load())/delta),
	)
	if *saveResults {
		outFile.Close()
	}
}
