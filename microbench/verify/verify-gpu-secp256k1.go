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

// #cgo CFLAGS: -I../../accelerator/gpu_cuda
// #cgo LDFLAGS: -L../../accelerator/gpu_cuda -lacc-gpu
// #include "header_gpu.h"
import "C"

import (
	// "encoding/hex"
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
	keyFilePrefix = kingpin.Flag("key-file", "Prefix of key files").Required().String()	
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

	fmt.Printf("%x\n%x\n", pub.X, pub.Y)

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
			buf := make([]byte, len(operands[3]))		// operands[3] is the value
			copy(buf, operands[3])
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

	// init GPU - batch 128
	batch := 8192	
	cbatch := C.int(batch)
	C.init_gpu(cbatch)
	pkeys := make([]byte, 64 * batch)
	signatures := make([]byte, 64 * batch)
	messages := make([]byte, 1024 * batch)
	lengths := make([]C.int, batch)
	idx := 0
	nreq := 0


	fmt.Println("Start running ...")
	if *saveResults {
		outFile, err := os.Create("output-cpu.txt")
		if err != nil {
			fmt.Printf("Error creating output file: %v\n", err)
		}
		// TODO
		/*
		for pair := range runBuf {

			// TODO

			outFile.WriteString(string(pair.a) + "\n")
			outFile.WriteString(hex.EncodeToString(dig) + " " + hex.EncodeToString(pair.b[:64]) + " " + res + "\n")
			// outFile.WriteString(hex.EncodeToString(dig) + " " + res + "\n")
		}
		*/
		outFile.Close()
	} else {
		latency := 0.0
		batches := 0
		start := time.Now()
		
		for pair := range runBuf {
		copy(pkeys[idx*64:(idx+1)*64], pubkey[1:])
		copy(signatures[idx*64:(idx+1)*64], pair.b[:64])
		copy(messages[idx*1024:(idx+1)*1024], pair.a)
		lengths[idx] = C.int(len(pair.a))
		idx++
		if idx == batch {
			pkptr := (*C.uchar)(unsafe.Pointer(&pkeys[0]))
			mptr := (*C.uchar)(unsafe.Pointer(&messages[0]))
			sptr := (*C.uchar)(unsafe.Pointer(&signatures[0]))
			lptr := (*C.int)(unsafe.Pointer(&lengths[0]))
			start1 := time.Now()
			C.run_keccak_secp256k1(cbatch, pkptr, sptr, mptr, lptr)
			latency += time.Since(start1).Seconds()
			batches++
			idx = 0
			nreq += batch
		}
	}

		delta := time.Since(start).Seconds()
		fmt.Printf("Throughput on GPU to handle %v requests: %v req/s\n",
			nreq, int64(float64(nreq)/delta),
		)
		fmt.Printf("Latency per batch: %v s\n", latency/float64(batches));
	}
}
