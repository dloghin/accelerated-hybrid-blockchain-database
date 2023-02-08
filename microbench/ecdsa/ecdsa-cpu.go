package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"

	"gopkg.in/alecthomas/kingpin.v2"

	"hbdb/src/utils"
)

var (
	keyFilePrefix = kingpin.Flag("key-file", "Prefix of Key Files").Required().String()
	concurrency   = kingpin.Flag("nthreads", "Number of threads for each driver").Default("10").Int()
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

	N := 100000
	wg := &sync.WaitGroup{}
	start := time.Now()
	for k := 0; k < *concurrency; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < N; i++ {
				if !secp256k1.VerifySignature(pubkey, dig, signature[:64]) {
					fmt.Println("Invalid Signature")
				}
			}
		}()
	}
	wg.Wait()

	delta := time.Since(start).Seconds()
	fmt.Printf("Throughput with %v concurrency to handle %v requests: %v req/s\n",
		*concurrency, *concurrency*N, int64(float64(*concurrency*N)/delta),
	)
}
