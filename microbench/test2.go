package main

import (
	"encoding/hex"
	"fmt"
	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	msg := "abcdefghijklmnopqrstuvwxyz"
	hash := "2b2ec6a6dc88bcec9df0ca5231c4b4e45d6f298c9437228929395335ada0149d"
	bhash, _ := hex.DecodeString(hash)
	for i := 0; i < 32; i++ {
	    fmt.Printf("%d,", bhash[i])
	}
	fmt.Println()
	hash1 := crypto.Keccak256([]byte(msg))
	fmt.Printf("%s\n", hex.EncodeToString(hash1))
	hash2 := crypto.Keccak256(append(bhash[:], hash1[:]...))
	fmt.Printf("%s\n", hex.EncodeToString(hash2))
}
