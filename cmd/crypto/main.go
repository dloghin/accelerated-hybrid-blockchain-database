package main

import (
	"crypto/ecdsa"
	"encoding/hex"
	"fmt"
	"github.com/ethereum/go-ethereum/crypto"
	"io/ioutil"
)

func SaveECDSAPub(file string, key ecdsa.PublicKey) error {
	k := hex.EncodeToString(crypto.FromECDSAPub(&key))
	return ioutil.WriteFile(file, []byte(k), 0600)
}

func GenerateAndSave(filename string) {
	pvk, err := crypto.GenerateKey()
	if err != nil {
		fmt.Printf("Error %v\n", err)
		return
	}
	err = crypto.SaveECDSA(filename+".pvk", pvk)
	if err != nil {
		fmt.Printf("Error %v\n", err)
		return
	}
	err = SaveECDSAPub(filename+".pub", pvk.PublicKey)
	if err != nil {
		fmt.Printf("Error %v\n", err)
		return
	}
}

func main() {
	GenerateAndSave("client")
	GenerateAndSave("server1")
	GenerateAndSave("server2")
	GenerateAndSave("server3")
	GenerateAndSave("server4")
}
