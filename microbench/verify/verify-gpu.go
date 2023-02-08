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
// #cgo LDFLAGS: -L/home/dumi/git/RapidEC/src -lverify
// #include "hbdb_interface.h"

import (
	"fmt"
	"hash"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"go.uber.org/atomic"
	"golang.org/x/crypto/sha3"
	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"
	"hbdb/src/utils"
)

type Pair struct {
	a, b []byte
}

var (
	dataRun       = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
	keyFilePrefix = kingpin.Flag("key-file", "Prefix of Key Files").Required().String()
	concurrency   = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
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

func main() {
	kingpin.Parse()

	// load private key
	fmt.Println("Load keys ...")
	pvk, err := crypto.LoadECDSA(*keyFilePrefix + ".pvk")
	if err != nil {
		panic(err)
	}

	pub, err := utils.LoadECDSAPub(*keyFilePrefix + ".pub")
	if err != nil {
		panic(err)
	}
	pvk.PublicKey = *pub

	pubkey, privkey := utils.DecodeKeyPair(pvk)

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
			sig, err := secp256k1.Sign(msg, privkey)
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

	// init kernel
	num := 100
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
			var rptr *C.int
			rptr = C.get_results(cnum)
			if *saveResults {
				for i := 0; i < num; i++ {
					if rptr[i] == 1 {
						outFile.WriteString("1\n")
					} else {
						outFile.WriteString("0\n")
					}
				}
			}

			// reset counters
			idx = 0
			offset = 0
		}
	}
	delta := time.Since(start).Seconds()
	fmt.Printf("Throughput with %v concurrency to handle %v requests: %v req/s\n",
		*concurrency, reqNum, int64(float64(reqNum.Load())/delta),
	)
}
