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

package ledger

import (
	"fmt"
	"sync"

	"github.com/dgraph-io/badger/v3"
	"github.com/ethereum/go-ethereum/crypto"

	pbv "hbdb/proto/hbdb"
)

type LogLedger struct {
	prevBlockHash []byte
	store         *badger.DB
}

func NewLedger(ledgerPath string, withMerkleTree bool) (*LogLedger, error) {
	db, err := badger.Open(badger.DefaultOptions(ledgerPath))
	if err != nil {
		return nil, err
	}

	return &LogLedger{
		prevBlockHash: []byte("0x00"),
		store:         db,
	}, nil
}

// return blkId, blkHash, txnHash
func (l *LogLedger) ProveKey(key []byte) (string, string, string, error) {
	blkId := ""
	blkHash := ""
	txnHash := ""
	err := l.store.View(func(txn *badger.Txn) error {
		dbkey := fmt.Sprintf("key_%s_blk", string(key))
		item, err := txn.Get([]byte(dbkey))
		if err != nil {
			return err
		}
		var val []byte
		item.Value(func(v []byte) error {
			val = make([]byte, len(v))
			copy(val, v)
			return nil
		})
		blkId = string(val)
		dbkey = fmt.Sprintf("blk_%s_hash", blkId)
		item, err = txn.Get([]byte(dbkey))
		if err != nil {
			return err
		}
		item.Value(func(v []byte) error {
			val = make([]byte, len(v))
			copy(val, v)
			return nil
		})
		blkHash = string(val)
		dbkey = fmt.Sprintf("key_%s_hash", string(key))
		item, err = txn.Get([]byte(dbkey))
		if err != nil {
			return err
		}
		item.Value(func(v []byte) error {
			val = make([]byte, len(v))
			copy(val, v)
			return nil
		})
		txnHash = string(val)
		return nil
	})
	return blkId, blkHash, txnHash, err
}

func (l *LogLedger) AppendBlk(block pbv.Block, blockData []byte, lock *sync.Mutex) error {
	// run the steps in parallel
	wg := &sync.WaitGroup{}

	// Step 1 - Verify
	var txn_hash []byte
	var blk_hash []byte
	wg.Add(1)
	go func() {
		defer wg.Done()
		// txn_hash := accelerator.Keccak256(blockData, lock)
		txn_hash := crypto.Keccak256(blockData)
		blk_hash := crypto.Keccak256(append(l.prevBlockHash, txn_hash...))
		l.prevBlockHash = blk_hash
	}()

	// Step 2 - Save data with no dependency to Step 1
	err := l.store.Update(func(txn *badger.Txn) error {
		txn.Set([]byte("tip"), []byte(block.BlkId))
		key := fmt.Sprintf("blk_%s_txn", block.BlkId)
		txn.Set([]byte(key), blockData)
		for _, req := range block.Txs {
			key = fmt.Sprintf("txn_%s_blk", req.GetTxId())
			txn.Set([]byte(key), []byte(block.BlkId))
			key = fmt.Sprintf("key_%s_blk", req.GetKey())
			txn.Set([]byte(key), []byte(block.BlkId))
			key = fmt.Sprintf("key_%s_hash", req.GetKey())
			txn.Set([]byte(key), []byte(req.GetHash()))
		}
		return nil
	})

	wg.Wait()

	if err != nil {
		return err
	}

	// Step 3 - Save block hash and txn hash
	err = l.store.Update(func(txn *badger.Txn) error {
		key := fmt.Sprintf("blk_%s_hash", block.BlkId)
		txn.Set([]byte(key), (blk_hash))
		key = fmt.Sprintf("blk_%s_txn_hash", block.BlkId)
		txn.Set([]byte(key), []byte(txn_hash))
		return nil
	})
	return err
}

func (l *LogLedger) Close() error {
	return l.store.Close()
}
