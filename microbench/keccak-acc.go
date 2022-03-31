package main

// #cgo CFLAGS: -I../accelerator/sw/keccak256
// #cgo LDFLAGS: -L../accelerator/sw/keccak256 -lkeccak
// #include "keccak.h"
import "C"

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"

	"go.uber.org/atomic"
	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/benchmark"
)

var (
	dataRun     = kingpin.Flag("run-path", "Path of YCSB operation data").Required().String()
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

	// open output file if need to save
	var outFile *os.File
	if *saveResults {
		outFile, err = os.Create("output-acc.txt")
                if err != nil {
                        fmt.Printf("Error creating output file: %v\n", err)
                }
	}

	// init kernel
	xclpath := []byte("../accelerator/bin/keccak256_kernel.xclbin")
        cstr := (*C.char)(unsafe.Pointer(&xclpath[0]))
        C.init_kernel(cstr, 100)
	num := 100
	cnum := C.int(num)
	data := make([]byte, num*256)
	sizes := make([]C.int, num)
	idx := 0
	offset := 0
	// run in batches
	start := time.Now()
	for msg := range runBuf {
		// align to 8 bytes 
		l := len(msg)
		if l % 8 != 0 {
			fmt.Println("Message is not aligned to 8 bytes!\n")
			return
		}
		// copy data
		sizes[idx] = C.int(l)
		copy(data[offset:], msg)
		offset += l
		// fmt.Printf("Data size: %v\n", sizes[idx])
		// fmt.Printf("Data: %v\n", string(data[idx]))
		idx++
		if idx == num {
			// call kernel
			dptr := (*C.uchar)(unsafe.Pointer(&data[0]))
		        sptr := (*C.int)(unsafe.Pointer(&sizes[0]))
		        size := C.send_data(dptr, sptr, cnum)
		        // fmt.Printf("Size %v Offset %d\n", size, offset)
		        C.run_kernel(cnum, C.int(size))
		        rptr := C.get_results(cnum)
		        res := C.GoBytes(unsafe.Pointer(rptr), 3200)
			if *saveResults {
			        for i := 0; i < num; i++ {
					outFile.WriteString(hex.EncodeToString(res[32*i:32*i+32]) + "\n")
				}
			}

			// reset counters
			idx = 0
			offset = 0
		}
	}
	delta := time.Since(start).Seconds()
        fmt.Printf("Throughput on FPGA to handle %v requests: %v req/s\n",
		reqNum, int64(float64(reqNum.Load())/delta),
        )

	if *saveResults {
		outFile.Close()
	}
}
