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

// #cgo LDFLAGS: -L/home/dumi/git/hbdb_ecdsa/microbench -lkeccak
// #include "keccak.h"
// import "C"

import (
	"encoding/hex"
	"fmt"
	"hash"
	"os"
	"strings"
	"sync"
	"time"

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
			buf := make([]byte, len(operands[3])) // operands[3] is the value
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

	fmt.Println("Start running ...")
	if *saveResults {
		outFile, err := os.Create("output-cpu.txt")
		if err != nil {
			fmt.Printf("Error creating output file: %v\n", err)
		}
		for pair := range runBuf {
			dig := myKeccak256(pair.a)
			res := "0"
			if secp256k1.VerifySignature(pubkey, dig, pair.b[:64]) {
				res = "1"
			}
			if res == "1" {
				outFile.WriteString(string(pair.a) + "\n")
				// outFile.WriteString(hex.EncodeToString(dig) + " " + hex.EncodeToString(pair.b[:64]) + " " + res + "\n")
				outFile.WriteString(hex.EncodeToString(pair.b[:64]) + "\n")
			}
		}
		outFile.Close()
	} else {
		start := time.Now()
		for j := 0; j < *concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for pair := range runBuf {
					dig := myKeccak256(pair.a)
					secp256k1.VerifySignature(pubkey, dig, pair.b[:64])
				}
			}()
		}
		wg.Wait()
		delta := time.Since(start).Seconds()
		fmt.Printf("Throughput with %v concurrency to handle %v requests: %v req/s\n",
			*concurrency, reqNum, int64(float64(reqNum.Load())/delta),
		)
	}
}
