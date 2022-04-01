/*
 * Copyright 2022 Dumitrel Loghin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"crypto/ecdsa"
	"encoding/hex"
	"fmt"
	"io/ioutil"

	"github.com/ethereum/go-ethereum/crypto"
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
