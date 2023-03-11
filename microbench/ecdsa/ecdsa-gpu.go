package main

// #cgo CFLAGS: -I../ecdsav2
// #cgo LDFLAGS: -L../ecdsav2 -lsecp256k1-gpu -lgmp
// #include "secp256k1_gpu.h"
import "C"

import (
	"fmt"
	"time"
	"os"
//	"encoding/hex"
//	"strings"
	"unsafe"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"

	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/utils"
)

var (
	keyFilePrefix = kingpin.Flag("key-file", "Prefix of Key Files").Required().String()
	saveResults   = kingpin.Flag("save", "Save digests to output-cpu.txt").Default("false").Bool()
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

	pubkey, privkey := utils.DecodeKeyPair(pvk)

	dig := crypto.Keccak256([]byte("abcdefghijklmnopqrstuvwxyz"))
	signature, err := secp256k1.Sign(dig, privkey)
	if err != nil {
		fmt.Println(err)
	}

	var outFile *os.File
	if *saveResults {
		outFile, err = os.Create("output-cpu.txt")
		if err != nil {
			fmt.Printf("Error creating output file: %v\n", err)
		}
	}

	// init GPU - batch 128
	batch := 8192
	N := batch * 50
	cbatch := C.int(batch)
	C.init_gpu(cbatch)
	pkeys := make([]byte, 64 * batch)
	digests := make([]byte, 32 * batch)
	signatures := make([]byte, 64 * batch)
	idx := 0

	// pkx := strings.ToUpper(pvk.PublicKey.X.Text(16))
	// pky := strings.ToUpper(pvk.PublicKey.Y.Text(16))
	// pkx := pvk.PublicKey.X.Text(16)
	// pky := pvk.PublicKey.Y.Text(16)
	// fmt.Println(pkx)
	// fmt.Println(pky)
	// fmt.Println(hex.EncodeToString(dig))
	// fmt.Println(hex.EncodeToString(signature[:32]))
	// fmt.Println(hex.EncodeToString(signature[32:64]))

	start := time.Now()
	for i := 0; i < N; i++ {
		copy(pkeys[idx*64:(idx+1)*64], pubkey[1:])
		copy(digests[idx*32:(idx+1)*32], dig)
		copy(signatures[idx*64:(idx+1)*64], signature[:64])
		idx++
		if idx == batch {
			pkptr := (*C.uchar)(unsafe.Pointer(&pkeys[0]))
			dptr := (*C.uchar)(unsafe.Pointer(&digests[0]))
			sptr := (*C.uchar)(unsafe.Pointer(&signatures[0]))
			C.run_kernel(cbatch, pkptr, dptr, sptr)
			// fmt.Printf("Size %v Offset %d\n", size, offset)
			// res := C.GoBytes(unsafe.Pointer(C.run_kernel(cbatch, pkptr, dptr, sptr)), cbatch);
			// for j := 0; j < batch; j++ {
			//	fmt.Printf("%d ", res[j])
			// 	if res[j] != 1 {
			// 		fmt.Println("Invalid verification!")
			// 	}
			// }
			// fmt.Println()
			/*
			if *saveResults {
				px := strings.ToUpper(pvk.PublicKey.X.Text(16))
				py := strings.ToUpper(pvk.PublicKey.Y.Text(16))
				for j := 0; j < batch; j++ {
					outFile.WriteString(px + " " + py + " " + strings.ToUpper(hex.EncodeToString(digests[j*32:(j+1)*32])) + " " + strings.ToUpper(hex.EncodeToString(signatures[j*64:(j+1)*64])) + "\n")
				}
			}
			*/
			idx = 0
		}
	}
	delta := time.Since(start).Seconds()
	fmt.Printf("Throughput to handle %v requests: %v req/s\n",
		N, int64(float64(N)/delta),
	)

	if *saveResults {
		outFile.Close();
	}

	C.free_gpu();
}
