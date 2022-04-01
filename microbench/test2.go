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
