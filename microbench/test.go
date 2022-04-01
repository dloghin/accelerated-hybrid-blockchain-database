package main

import (
	"fmt"

	"encoding/hex"

	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("a"))))
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("ab"))))
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("abc"))))
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("abcd"))))
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("abcde"))))
	fmt.Printf("%s\n", hex.EncodeToString(crypto.Keccak256([]byte("abcdef"))))
}
