package main

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"
	"hbdb/src/utils"
)

var (
	dataRun       = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
	keyFilePrefix = kingpin.Flag("key-file-prefix", "ECDSA key file prefix").Required().String()
	concurrency   = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
	saveResults   = kingpin.Flag("save", "Save digests to output-cpu.txt").Bool()
)

func main() {
	kingpin.Parse()

	// load private key
	pvk, err := crypto.LoadECDSA(*keyFilePrefix + ".pvk")
	if err != nil {
		panic(err)
	}
	pubkey, privkey := utils.DecodeKeyPair(pvk)

	fmt.Println("Load data ...")
	runFile, err := os.Open(*dataRun)
	if err != nil {
		panic(err)
	}
	defer runFile.Close()
	runBuf := make(chan string, 100000)
	var reqNum int32
	reqNum = 0
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(runBuf)
		if err := benchmark.LineByLine(runFile, func(line string) error {
			atomic.AddInt32(&reqNum, 1)
			operands := strings.SplitN(line, " ", 5)
			l := len(operands[2])
			if l%8 != 0 {
				l = 8 * (l/8 + 1)
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
	wg.Wait()

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

		// Keccak256
		hashes := make([][]byte, atomic.LoadInt32(&reqNum))
		var reqIdx int32
		reqIdx = -1
		for j := 0; j < *concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for msg := range runBuf {
					cIdx := atomic.AddInt32(&reqIdx, 1)
					hashes[cIdx] = crypto.Keccak256([]byte(msg))
				}
			}()
		}

		start := time.Now()
		wg.Wait()
		delta := time.Since(start).Seconds()
		fmt.Printf("Time of Keccak with %v concurrency to handle %v requests: %v s\n", *concurrency, reqNum, delta)
		fmt.Printf("Throughput of Keccak with %v concurrency to handle %v requests: %v req/s\n",
			*concurrency, reqNum, int64(float64(atomic.LoadInt32(&reqNum))/delta),
		)

		// ECDSA secp256k1 sign
		signatures := make([][]byte, atomic.LoadInt32(&reqNum))
		reqIdx = -1
		for j := 0; j < *concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for atomic.LoadInt32(&reqIdx) < atomic.LoadInt32(&reqNum)-1 {
					cIdx := atomic.AddInt32(&reqIdx, 1)
					if cIdx >= atomic.LoadInt32(&reqNum) {
						break
					}
					signatures[cIdx], _ = secp256k1.Sign(hashes[cIdx], privkey)
				}
			}()
		}
		start = time.Now()
		wg.Wait()
		delta = time.Since(start).Seconds()
		fmt.Printf("Time of ECDSA Sign with %v concurrency to handle %v requests: %v s\n", *concurrency, reqNum, delta)
		fmt.Printf("Throughput of ECDSA Sign with %v concurrency to handle %v requests: %v req/s\n",
			*concurrency, reqNum, int64(float64(atomic.LoadInt32(&reqNum))/delta),
		)

		// ECDSA secp256k1 verify
		reqIdx = -1
		for j := 0; j < *concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for atomic.LoadInt32(&reqIdx) < atomic.LoadInt32(&reqNum)-1 {
					cIdx := atomic.AddInt32(&reqIdx, 1)
					if cIdx >= atomic.LoadInt32(&reqNum) {
						break
					}
					secp256k1.VerifySignature(pubkey, hashes[cIdx], signatures[cIdx])
				}
			}()
		}
		start = time.Now()
		wg.Wait()
		delta = time.Since(start).Seconds()
		fmt.Printf("Time of ECDSA Verify with %v concurrency to handle %v requests: %v s\n", *concurrency, reqNum, delta)
		fmt.Printf("Throughput of ECDSA Verify with %v concurrency to handle %v requests: %v req/s\n",
			*concurrency, reqNum, int64(float64(atomic.LoadInt32(&reqNum))/delta),
		)

	}
}
