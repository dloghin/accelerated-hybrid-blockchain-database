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

	"go.uber.org/atomic"
	"golang.org/x/crypto/sha3"
	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"
)

var (
	dataRun     = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
	concurrency = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
	saveResults = kingpin.Flag("save", "Save digests to output-cpu.txt").Bool()
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

	fmt.Println("Load data ...")
	runFile, err := os.Open(*dataRun)
	if err != nil {
		panic(err)
	}
	defer runFile.Close()
	runBuf := make(chan string, 100000)
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
			runBuf <- string(buf)
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
		for msg := range runBuf {
			hash := myKeccak256([]byte(msg))
			// outFile.WriteString("|" + msg + "|\n")
			// outFile.WriteString(hex.EncodeToString(hash) + " " + strconv.FormatInt(int64(len(msg)), 10) + "\n")
			outFile.WriteString(hex.EncodeToString(hash) + "\n")
		}
		outFile.Close()
	} else {
		for j := 0; j < *concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for msg := range runBuf {
					myKeccak256([]byte(msg))
				}
			}()
		}

		start := time.Now()
		wg.Wait()
		delta := time.Since(start).Seconds()
		fmt.Printf("Throughput with %v concurrency to handle %v requests: %v req/s\n",
			*concurrency, reqNum, int64(float64(reqNum.Load())/delta),
		)
	}
}
