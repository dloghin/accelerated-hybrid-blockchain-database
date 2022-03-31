package main

// #cgo LDFLAGS: -L/home/dumi/git/hbdb_ecdsa/microbench -lkeccak
// #include "keccak.h"
import "C"

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"go.uber.org/atomic"
	"gopkg.in/alecthomas/kingpin.v2"
	"github.com/ethereum/go-ethereum/crypto"

	"hbdb/src/benchmark"
)

var (
	dataRun     = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
	concurrency = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
	saveResults = kingpin.Flag("save", "Save digests to output-cpu.txt").Bool()
)

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
	                if l % 8 != 0 {
		                l = 8 * (l / 8 + 1)
			}
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
			hash := crypto.Keccak256([]byte(msg))
			outFile.WriteString(hex.EncodeToString(hash) + "\n")
		}
		outFile.Close()
	} else {
		for j := 0; j < *concurrency; j++ {
                        wg.Add(1)
                        go func() {
                                defer wg.Done()
                                for msg := range runBuf {
                                        crypto.Keccak256([]byte(msg))
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
