package ledger

import (
	"fmt"

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

func (l *LogLedger) AppendBlk(block pbv.Block, blockData []byte) error {
	txn_hash := crypto.Keccak256(blockData)
	blk_hash := crypto.Keccak256(append(txn_hash, l.prevBlockHash...))

	return l.store.Update(func(txn *badger.Txn) error {
		txn.Set([]byte("tip"), []byte(block.BlkId))
		key := fmt.Sprintf("blk_%s_hash", block.BlkId)
		txn.Set([]byte(key), []byte(blk_hash))
		key = fmt.Sprintf("blk_%s_txn_hash", block.BlkId)
		txn.Set([]byte(key), []byte(txn_hash))
		key = fmt.Sprintf("blk_%s_txn", block.BlkId)
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
}

func (l *LogLedger) Close() error {
	return l.store.Close()
}
