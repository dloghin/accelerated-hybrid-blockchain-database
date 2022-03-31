package server

// #cgo CFLAGS: -I../../accelerator/sw/keccak256
// #cgo LDFLAGS: -L../../accelerator/sw/keccak256 -lkeccak
// #include "keccak.h"
import "C"

import (
	"fmt"
	pbv "hbdb/proto/hbdb"
	"unsafe"
)

const MaxSize = 128

func InitAccelerator(xclbinpath string, batchSize int) {
	xclpath := []byte(xclbinpath)
	cstr := (*C.char)(unsafe.Pointer(&xclpath[0]))
	num := C.int(batchSize)
	C.init_kernel(cstr, num)
}

func VerifySignatureBatchAccelerator(setRequests *pbv.BatchSetRequest, getRequests *pbv.BatchGetRequest) ([]byte, []bool) {
	batchSize := 0
	if setRequests == nil {
		batchSize = len(getRequests.GetRequests())
	} else {
		batchSize = len(setRequests.GetRequests())
	}
	num := C.int(batchSize)
	data := make([]byte, batchSize*MaxSize)
	sizes := make([]int, batchSize)
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
		sizes[idx] = len(payload)
		copy(data[offset:], payload[:])
		offset += sizes[idx]
	}
	dptr := (*C.uchar)(unsafe.Pointer(&data[0]))
	sptr := (*C.int)(unsafe.Pointer(&sizes[0]))
	size := C.send_data(dptr, sptr, num)
	fmt.Printf("Size %v\n", size)
	C.run_kernel(size, num)
	rptr := C.get_results(num)
	hashes := C.GoBytes(unsafe.Pointer(rptr), 32*num)
	/*
		for i := 0; i < num; i++ {
			fmt.Printf("%v\n", hex.EncodeToString(res[32*i:32*i+31]))
		}
	*/
	valid := make([]bool, batchSize)
	for idx := 0; idx < batchSize; idx++ {
		valid[idx] = true
	}

	return hashes, valid
}
