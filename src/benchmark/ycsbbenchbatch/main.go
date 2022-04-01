package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto"

	"go.uber.org/atomic"

	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"
	"hbdb/src/driver"
	"hbdb/src/utils"
)

var (
	dataLoad          = kingpin.Flag("load-path", "Path of YCSB initial data").Required().String()
	dataRun           = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
	driverNum         = kingpin.Flag("ndrivers", "Number of drivers for sending requests").Default("4").Int()
	driverConcurrency = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
	serverAddrs       = kingpin.Flag("server-addrs", "Address of HBDB server nodes").Required().String()
	keyFilePrefix     = kingpin.Flag("key-file-prefix", "ECDSA key file prefix").Required().String()
	batchSize         = kingpin.Flag("batch-size", "Request batch size").Default("100").Int()
)

func main() {
	kingpin.Parse()

	// load private key
	pvk, err := crypto.LoadECDSA(*keyFilePrefix + ".pvk")
	if err != nil {
		panic(err)
	}

	pub, err := utils.LoadECDSAPub(*keyFilePrefix + ".pub")
	if err != nil {
		panic(err)
	}
	pvk.PublicKey = *pub

	addrs := strings.Split(*serverAddrs, ",")
	clis := make([]*driver.Driver, 0)
	defer func() {
		for _, cli := range clis {
			cli.Close()
		}
	}()
	for i := 0; i < *driverNum; i++ {
		cli, err := driver.Open(addrs[i%len(addrs)], pvk)
		if err != nil {
			panic(err)
		}
		clis = append(clis, cli)
	}

	fmt.Println("Start loading ...")
	reqNum := atomic.NewInt64(0)

	loadFile, err := os.Open(*dataLoad)
	if err != nil {
		panic(err)
	}
	defer loadFile.Close()
	loadBuf := make(chan [3]string, 10)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(loadBuf)
		if err := benchmark.LineByLine(loadFile, func(line string) error {
			operands := strings.SplitN(line, " ", 5)
			loadBuf <- [3]string{operands[2], operands[3], operands[4]}
			return nil
		}); err != nil {
			panic(err)
		}
	}()
	latencyCh := make(chan time.Duration, 1024)
	wg2 := sync.WaitGroup{}
	wg2.Add(1)
	var avaLatency float64
	go func() {
		defer wg2.Done()
		all := int64(0)
		for ts := range latencyCh {
			all += ts.Microseconds()
		}
		avaLatency = float64(all) / (1000 * float64(reqNum.Load()))
	}()
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for kv := range loadBuf {
				ver, err := strconv.ParseInt(kv[2], 10, 64)
				if err != nil {
					panic(err)
				}
				clis[0].Set(context.Background(), kv[0], kv[1], ver)
			}
		}()
	}
	wg.Wait()
	fmt.Println("End loading ...")

	fmt.Println("Start running ...")
	runFile, err := os.Open(*dataRun)
	if err != nil {
		panic(err)
	}
	defer runFile.Close()
	runBuf := make(chan *benchmark.BatchRequest, (*driverNum)*(*driverConcurrency))
	getReq := &benchmark.BatchRequest{
		ReqType: benchmark.GetOp,
		Keys:    make([]string, *batchSize),
	}
	getReqIdx := 0
	setReq := &benchmark.BatchRequest{
		ReqType:  benchmark.SetOp,
		Keys:     make([]string, *batchSize),
		Vals:     make([]string, *batchSize),
		Versions: make([]int64, *batchSize),
	}
	setReqIdx := 0
	var lastSetKey string
	var lastSetVer int64
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(runBuf)
		if err := benchmark.LineByLine(runFile, func(line string) error {
			operands := strings.SplitN(line, " ", 5)
			ver, err := strconv.ParseInt(operands[4], 10, 64)
			if err != nil {
				panic(err)
			}
			if operands[0] == "READ" {
				getReq.Keys[getReqIdx] = operands[2]
				getReqIdx++
			} else {
				setReq.Keys[setReqIdx] = operands[2]
				setReq.Vals[setReqIdx] = operands[3]
				setReq.Versions[setReqIdx] = ver
				setReqIdx++
			}
			if getReqIdx == *batchSize {
				runBuf <- getReq
				reqNum.Add(int64(*batchSize))
				getReqIdx = 0
				getReq = &benchmark.BatchRequest{
					ReqType: benchmark.GetOp,
					Keys:    make([]string, *batchSize),
				}
			}
			if setReqIdx == *batchSize {
				lastSetKey = setReq.Keys[setReqIdx-1]
				lastSetVer = setReq.Versions[setReqIdx-1]
				runBuf <- setReq
				reqNum.Add(int64(*batchSize))
				setReqIdx = 0
				setReq = &benchmark.BatchRequest{
					ReqType:  benchmark.SetOp,
					Keys:     make([]string, *batchSize),
					Vals:     make([]string, *batchSize),
					Versions: make([]int64, *batchSize),
				}
			}
			return nil
		}); err != nil {
			panic(err)
		}
	}()
	time.Sleep(5 * time.Second)

	for i := 0; i < *driverNum; i++ {
		for j := 0; j < *driverConcurrency; j++ {
			wg.Add(1)
			go func(seq int) {
				defer wg.Done()
				for op := range runBuf {
					switch op.ReqType {
					case benchmark.GetOp:
						start := time.Now()
						clis[seq].BatchGet(context.Background(), *batchSize, op.Keys)
						latencyCh <- time.Since(start)
					case benchmark.SetOp:
						start := time.Now()
						clis[seq].BatchSet(context.Background(), *batchSize, op.Keys, op.Vals, op.Versions)
						latencyCh <- time.Since(start)
					default:
						panic(fmt.Sprintf("invalid operation: %v", op.ReqType))
					}
				}
			}(i)
		}
	}
	start := time.Now()
	wg.Wait()
	close(latencyCh)
	wg2.Wait()
	fmt.Println("Wait for last Set to take effect ...")
	for {
		_, ver, err := clis[0].Get(context.Background(), lastSetKey)
		if err != nil {
			fmt.Printf("Error in Get: %v\n", err)
		} else {
			if ver == lastSetVer+1 {
				break
			}
		}
	}
	delta := time.Since(start).Seconds()
	fmt.Printf("Throughput of %v drivers with %v concurrency to handle %v requests: %v req/s\n",
		*driverNum, *driverConcurrency, reqNum,
		int64(float64(reqNum.Load())/delta),
	)
	fmt.Printf("Average latency: %v ms\n", avaLatency)
	fmt.Printf("Batch size: %d\n", *batchSize)
}
