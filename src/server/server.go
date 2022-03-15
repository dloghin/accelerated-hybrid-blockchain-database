package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"encoding/hex"

	"go.uber.org/atomic"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
	"gopkg.in/confluentinc/confluent-kafka-go.v1/kafka"

	pbv "hbdb/proto/hbdb"
	"hbdb/src/ledger"
)

type server struct {
	ctx     context.Context
	cancel  context.CancelFunc
	config  *Config
	l       *ledger.LogLedger
	blkcnt  *atomic.Int64
	cli     *redis.Client
	puller  *kafka.Consumer
	pusher  *kafka.Producer
	msgCh   chan *pbv.SharedLog
	txMsgCh chan string
}

type BlockPurpose struct {
	blk      *pbv.Block
	approved map[string]struct{}
}

func NewServer(redisCli *redis.Client, consumer *kafka.Consumer, producer *kafka.Producer, ledgerPath string, config *Config) *server {
	ctx, cancel := context.WithCancel(context.Background())
	l, err := ledger.NewLedger(ledgerPath, true)
	if err != nil {
		log.Fatalf("Create ledger failed: %v", err)
	}
	s := &server{
		ctx:     ctx,
		cancel:  cancel,
		l:       l,
		blkcnt:  atomic.NewInt64(0),
		config:  config,
		cli:     redisCli,
		puller:  consumer,
		pusher:  producer,
		msgCh:   make(chan *pbv.SharedLog, 10000),
		txMsgCh: make(chan string, 10000),
	}
	if err := s.puller.Subscribe(config.Topic, nil); err != nil {
		log.Fatalf("Subscribe topic %v failed: %v", config.Topic, err)
	}
	go s.applyLoop()
	go s.batchLoop()
	return s
}

func (s *server) verifyAndCommit(msg *kafka.Message, block pbv.Block) {
	for _, sl := range block.Txs {
		for _, t := range sl.Sets {
			res, err := s.cli.Get(s.ctx, t.GetKey()).Result()
			if err != nil && err != redis.Nil {
				log.Fatalf("Commit log %v DB get failed: %v", block.GetBlkId(), err)
			}
			if err == nil {
				v, err := Decode(res)
				if err != nil {
					log.Fatalf("Commit log %v decode failed: %v", block.GetBlkId(), err)
				}
				if v.Version > t.Version {
					log.Printf("Abort transaction in block %v for key %s local version %d request version %d\n", block.GetBlkId(), t.GetKey(), v.Version, t.Version)
					continue
				}
			}
			entry, err := Encode(t.GetValue(), t.GetVersion()+1)
			if err != nil {
				log.Fatalf("Commit log %v encode failed: %v", block.GetBlkId(), err)
			}
			if err := s.cli.Set(s.ctx, t.GetKey(), entry, 0).Err(); err != nil {
				log.Fatalf("Commit log %v redis set failed: %v", block.GetBlkId(), err)
			}
			s.txMsgCh <- t.GetTxId()
		}
	}
	s.l.AppendBlk(msg.Value) // avoid remarshalling from blkBuf.blk
}

func (s *server) applyLoop() {
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}
		msg, err := s.puller.ReadMessage(-1)
		if err != nil {
			log.Fatalf("Consumer read msg failed: %v", err)
		}
		var blk pbv.Block
		if err := proto.Unmarshal(msg.Value, &blk); err != nil {
			log.Fatalf("Parse msg failed: %v", err)
		}
		switch blk.GetType() {
		case pbv.MessageType_Approve:
			s.verifyAndCommit(msg, blk)
		default:
			log.Fatalf("Invalid shared log type: %v", blk.GetType())
		}
	}
}

func (s *server) sendBlock(block *pbv.Block) {
	block.BlkId = fmt.Sprintf("%s_%d", s.config.Signature, s.blkcnt.Load())
	s.blkcnt.Add(1)
	blkLog, err := proto.Marshal(block)
	if err != nil {
		log.Fatalf("Block log failed: %v", err)
	}
	if err := s.pusher.Produce(&kafka.Message{
		TopicPartition: kafka.TopicPartition{Topic: &s.config.Topic, Partition: kafka.PartitionAny},
		Value:          blkLog,
	}, nil); err != nil {
		log.Fatalf("%s produce block log failded: %v", s.config.Signature, err)
	}
	// empty block for next round
	block.Txs = make([]*pbv.SharedLog, 0)
}

func (s *server) batchLoop() {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	block := &pbv.Block{
		Txs:       make([]*pbv.SharedLog, 0),
		Type:      pbv.MessageType_Approve,
		Signature: s.config.Signature,
	}
	defer close(s.msgCh)
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-t.C:
			if len(block.Txs) > 0 {
				s.sendBlock(block)
			}
		case txn := <-s.msgCh:
			block.Txs = append(block.Txs, txn)
			if len(block.Txs) >= s.config.BlockSize {
				s.sendBlock(block)
			}
		}
	}
}

func (s *server) Get(ctx context.Context, req *pbv.GetRequest) (*pbv.GetResponse, error) {
	// check signature
	/*
		if len(req.GetPubkey()) == 0 {
			fmt.Printf("Public key: %v\n", req.GetPubkey())
			fmt.Printf("Signature: %v\n", req.GetSignature())
			fmt.Printf("Key: %v\n", req.GetKey())
			return nil, errors.New("Invalid key")
		}
	*/
	pubkey, err := hex.DecodeString(req.GetPubkey())
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return nil, err
	}
	signature, err := hex.DecodeString(req.GetSignature())
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return nil, err
	}
	dig := crypto.Keccak256([]byte(req.GetKey()))
	if len(dig) != 32 {
		fmt.Println("Error len digest != 32")
		return nil, nil
	}
	if !secp256k1.VerifySignature(pubkey, dig, signature[:64]) {
		fmt.Println("Error in VerifySignature Get")
		return nil, errors.New("Invalid Signature")
	}

	// fetch data
	res, err := s.cli.Get(ctx, req.GetKey()).Result()
	if err != nil {
		return nil, err
	}
	v, err := Decode(res)
	if err != nil {
		return nil, err
	}

	return &pbv.GetResponse{Value: v.Val, Version: v.Version}, nil
}

func (s *server) Set(ctx context.Context, req *pbv.SetRequest) (*pbv.SetResponse, error) {
	// check signature
	pubkey, err := hex.DecodeString(req.GetPubkey())
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return nil, err
	}
	signature, err := hex.DecodeString(req.GetSignature())
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return nil, err
	}
	data := fmt.Sprintf("%s%s%d", req.GetKey(), req.GetValue(), req.GetVersion())
	dig := crypto.Keccak256([]byte(data))
	if !secp256k1.VerifySignature(pubkey, dig, signature[:64]) {
		fmt.Println("Error in VerifySignature Set")
		return nil, errors.New("Invalid Signature")
	}

	// check version
	dbrecord, err := s.cli.Get(ctx, req.GetKey()).Result()
	if err != nil {
		return nil, err
	}
	record, err := Decode(dbrecord)
	if err != nil {
		return nil, err
	}
	if record != nil && record.Version > req.GetVersion() {
		return &pbv.SetResponse{}, errors.New("Rejected (wrong version)")
	}
	// prepare request
	t := time.Now().Unix()
	txId := fmt.Sprintf("_txid#%s#%d", s.config.Signature, t)
	sets := []*pbv.SetRequest{{
		Signature: req.GetSignature(),
		Key:       req.GetKey(),
		Value:     req.GetValue(),
		Version:   req.GetVersion(),
		TxId:      txId,
	}}
	s.msgCh <- &pbv.SharedLog{
		Seq:  req.GetVersion(),
		Sets: sets,
	}
	// wait for it to be committed or aborted
	go func() {
		for {
			msg := <-s.txMsgCh
			if msg == txId {
				break
			}
		}
	}()

	return &pbv.SetResponse{Txid: txId}, nil
}

func (s *server) Verify(ctx context.Context, req *pbv.VerifyRequest) (*pbv.VerifyResponse, error) {
	proof, err := s.l.ProveKey([]byte(req.GetKey()))
	if err != nil {
		return nil, err
	}
	return &pbv.VerifyResponse{
		RootDigest:            s.l.GetRootDigest(),
		SideNodes:             proof.SideNodes,
		NonMembershipLeafData: proof.NonMembershipLeafData,
	}, nil
}

func (s *server) BatchGet(ctx context.Context, requests *pbv.BatchGetRequest) (*pbv.BatchGetResponse, error) {
	responses := make([]*pbv.GetResponse, len(requests.GetRequests()))
	for idx, req := range requests.GetRequests() {
		res, _ := s.Get(ctx, req)
		responses[idx] = res
	}
	return &pbv.BatchGetResponse{Responses: responses}, nil
}

func (s *server) BatchSet(ctx context.Context, requests *pbv.BatchSetRequest) (*pbv.BatchSetResponse, error) {
	responses := make([]*pbv.SetResponse, len(requests.GetRequests()))
	for idx, req := range requests.GetRequests() {
		res, _ := s.Set(ctx, req)
		responses[idx] = res
	}
	return &pbv.BatchSetResponse{Responses: responses}, nil
}
