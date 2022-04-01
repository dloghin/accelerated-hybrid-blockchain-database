package accelerator

// #cgo CFLAGS: -I../../accelerator/sw/keccak256
// #cgo LDFLAGS: -L../../accelerator/sw/keccak256 -lkeccak
// #include "keccak.h"
import "C"

import (
	"fmt"
	pbv "hbdb/proto/hbdb"
	"sync"
	"time"
	"unsafe"
)

const MaxSize = 256

func InitAccelerator(xclbinpath string, batchSize int) {
	xclpath := []byte(xclbinpath)
	cstr := (*C.char)(unsafe.Pointer(&xclpath[0]))
	num := C.int(batchSize)
	C.init_kernel(cstr, num)
}

func VerifySignatureBatchAccelerator(setRequests *pbv.BatchSetRequest, getRequests *pbv.BatchGetRequest, lock *sync.Mutex) ([]byte, []bool) {
	// protect the call with a lock
	lock.Lock()
	defer lock.Unlock()

	start := time.Now()
	// prepare data
	batchSize := 0
	if setRequests == nil {
		batchSize = len(getRequests.GetRequests())
	} else {
		batchSize = len(setRequests.GetRequests())
	}
	num := C.int(batchSize)
	data := make([]byte, batchSize*MaxSize)
	sizes := make([]C.int, batchSize)
	offset := 0
	for idx := 0; idx < batchSize; idx++ {
		payload := ""
		if setRequests == nil {
			req := getRequests.GetRequests()[idx]
			payload = req.GetKey()
		} else {
			req := setRequests.GetRequests()[idx]
			payload = fmt.Sprintf("%s%s%d", req.GetKey(), req.GetValue(), req.GetVersion())
		}

		l := len(payload)
		if l%8 != 0 {
			l = 8 * (l/8 + 1)
		}
		buf := make([]byte, l)
		copy(buf, []byte(payload))
		copy(data[offset:], buf)
		offset += l
		sizes[idx] = C.int(l)
	}
	dptr := (*C.uchar)(unsafe.Pointer(&data[0]))
	sptr := (*C.int)(unsafe.Pointer(&sizes[0]))
	size := C.send_data(dptr, sptr, num)
	// fmt.Printf("Size %v\n", size)

	// call kernel
	C.run_kernel(num, C.int(size))

	// get digests (hashes)
	rptr := C.get_results(num)
	hashes := C.GoBytes(unsafe.Pointer(rptr), 32*num)
	delta := time.Since(start)
	// fmt.Printf("FPGA phase took %f s\n", delta.Seconds())

	// ECDSA validation (fake)
	time.Sleep(delta / 2)
	valid := make([]bool, batchSize)
	for idx := 0; idx < batchSize; idx++ {
		valid[idx] = true
	}

	return hashes, valid
}

func Keccak256(data []byte, lock *sync.Mutex) []byte {
	// protect the call with a lock
	lock.Lock()
	defer lock.Unlock()

	// prepare data
	num := C.int(1)
	sizes := make([]C.int, 1)
	sizes[0] = C.int(len(data))
	dptr := (*C.uchar)(unsafe.Pointer(&data[0]))
	sptr := (*C.int)(unsafe.Pointer(&sizes[0]))
	size := C.send_data(dptr, sptr, num)

	// call kernel
	C.run_kernel(num, C.int(size))

	// get digests (hashes)
	rptr := C.get_results(num)
	hash := C.GoBytes(unsafe.Pointer(rptr), 32*num)

	return hash
}
