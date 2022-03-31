package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"go.uber.org/atomic"

	"github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
	"github.com/gookit/event"
	"gopkg.in/confluentinc/confluent-kafka-go.v1/kafka"

	pbv "hbdb/proto/hbdb"
	"hbdb/src/ledger"
)

type server struct {
	ctx          context.Context
	cancel       context.CancelFunc
	config       *Config
	ledger       *ledger.LogLedger
	blkcnt       *atomic.Int64
	txcnt        *atomic.Int64
	localDB      *redis.Client
	puller       *kafka.Consumer
	pusher       *kafka.Producer
	setRequestCh chan *pbv.SetRequest
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
		ctx:          ctx,
		cancel:       cancel,
		ledger:       l,
		blkcnt:       atomic.NewInt64(0),
		txcnt:        atomic.NewInt64(0),
		config:       config,
		localDB:      redisCli,
		puller:       consumer,
		pusher:       producer,
		setRequestCh: make(chan *pbv.SetRequest, 50000),
	}
	if err := s.puller.Subscribe(config.Topic, nil); err != nil {
		log.Fatalf("Subscribe topic %v failed: %v", config.Topic, err)
	}
	if s.config.ParallelBatchProcessing {
		fmt.Println("Parallel batch request processing.")
	}
	go s.applyLoop()
	go s.batchLoop()
	return s
}

func (s *server) verifyAndCommit(msg *kafka.Message, block pbv.Block) {
	for _, req := range block.Txs {
		res, err := s.localDB.Get(s.ctx, req.GetKey()).Result()
		if err != nil && err != redis.Nil {
			log.Fatalf("Commit log %v DB get failed: %v", block.GetBlkId(), err)
		}
		if err == nil {
			v, err := Decode(res)
			if err != nil {
				log.Fatalf("Commit log %v decode failed: %v", block.GetBlkId(), err)
			}
			if v.Version > req.Version {
				log.Printf("Abort transaction in block %v for key %s local version %d request version %d\n",
					block.GetBlkId(), req.GetKey(), v.Version, req.Version)
				// s.setDoneCh <- req.GetTxId()
				event.MustFire(req.TxId, nil)
				continue
			}
		}
		entry, err := Encode(req.GetValue(), req.GetVersion()+1)
		if err != nil {
			log.Fatalf("Commit log %v encode failed: %v", block.GetBlkId(), err)
		}
		if err := s.localDB.Set(s.ctx, req.GetKey(), entry, 0).Err(); err != nil {
			log.Fatalf("Commit log %v redis set failed: %v", block.GetBlkId(), err)
		}
		// s.setDoneCh <- req.GetTxId()
		event.MustFire(req.TxId, nil)
	}
	s.ledger.AppendBlk(block, msg.Value) // avoid remarshalling from blkBuf.blk
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
		s.verifyAndCommit(msg, blk)
	}
}

func (s *server) sendBlock(block *pbv.Block) {
	block.BlkId = fmt.Sprintf("%s_%d", s.config.Signature, s.blkcnt.Load())
	s.blkcnt.Add(1)
	block.Size = int64(len(block.Txs))
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
	block.Txs = make([]*pbv.SetRequest, 0)
}

func (s *server) batchLoop() {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	block := &pbv.Block{
		Txs: make([]*pbv.SetRequest, 0),
	}
	defer close(s.setRequestCh)
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-t.C:
			if len(block.Txs) > 0 {
				s.sendBlock(block)
			}
		case txn := <-s.setRequestCh:
			block.Txs = append(block.Txs, txn)
			if len(block.Txs) >= s.config.BlockSize {
				s.sendBlock(block)
			}
		}
	}
}

func (s *server) getLocalData(ctx context.Context, key string) (*pbv.GetResponse, error) {
	res, err := s.localDB.Get(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	decoded, err := Decode(res)
	if err != nil {
		return nil, err
	}
	return &pbv.GetResponse{Value: decoded.Val, Version: decoded.Version}, nil
}

func (s *server) Get(ctx context.Context, req *pbv.GetRequest) (*pbv.GetResponse, error) {
	// fetch data
	return s.getLocalData(ctx, req.GetKey())
}

func (s *server) checkMVCC(ctx context.Context, key string, version int64) bool {
	dbrecord, err := s.localDB.Get(ctx, key).Result()
	if err == redis.Nil {
		// if there is no such record, we just accept the request
		return true
	}
	if err != nil {
		fmt.Printf("Error in checkMVCC Get local DB: %v\n", err)
		return false
	}
	record, err := Decode(dbrecord)
	if err != nil {
		fmt.Printf("Error in checkMVCC decode: %v\n", err)
		return false
	}
	if record != nil && record.Version > version {
		fmt.Printf("Rejected MVCC for key %s request version %d local version %d\n", key, version, record.Version)
		return false
	}
	return true
}

func (s *server) prepareSetRequest(req *pbv.SetRequest) *pbv.SetRequest {
	req.TxId = fmt.Sprintf("txid%stx%d", s.config.Signature, s.txcnt.Load())
	s.txcnt.Inc()
	return req
}

func (s *server) SetSync(ctx context.Context, req *pbv.SetRequest) (*pbv.SetResponse, error) {
	// check version
	if !s.checkMVCC(ctx, req.GetKey(), req.GetVersion()) {
		return nil, errors.New("Invalid Request Version")
	}

	// prepare request
	req = s.prepareSetRequest(req)

	// wait for it to be committed or aborted
	wg := &sync.WaitGroup{}
	wg.Add(1)
	event.On(req.GetTxId(), event.ListenerFunc(func(e event.Event) error {
		defer wg.Done()
		fmt.Printf("Tx done %s\n", req.TxId)
		return nil
	}), event.Normal)

	// send and wait
	s.setRequestCh <- req
	wg.Wait()

	return &pbv.SetResponse{Txid: req.TxId}, nil
}

func (s *server) Set(ctx context.Context, req *pbv.SetRequest) (*pbv.SetResponse, error) {
	// check version
	if !s.checkMVCC(ctx, req.GetKey(), req.GetVersion()) {
		return nil, errors.New("Invalid Request Version")
	}

	// prepare and send request
	req = s.prepareSetRequest(req)
	s.setRequestCh <- req

	return &pbv.SetResponse{Txid: req.TxId}, nil
}

func (s *server) Verify(ctx context.Context, req *pbv.VerifyRequest) (*pbv.VerifyResponse, error) {
	blkId, blkHash, txnHash, err := s.ledger.ProveKey([]byte(req.GetKey()))
	if err != nil {
		return nil, err
	}
	return &pbv.VerifyResponse{
		BlockId:   blkId,
		BlockHash: blkHash,
		TxnHash:   txnHash,
	}, nil
}

func (s *server) BatchGet(ctx context.Context, requests *pbv.BatchGetRequest) (*pbv.BatchGetResponse, error) {
	responses := make([]*pbv.GetResponse, len(requests.GetRequests()))

	if s.config.ParallelBatchProcessing {
		wg := &sync.WaitGroup{}
		for i, r := range requests.GetRequests() {
			wg.Add(1)
			go func(idx int, req *pbv.GetRequest) {
				defer wg.Done()
				responses[idx], _ = s.getLocalData(ctx, req.GetKey())
			}(i, r)
		}
		wg.Wait()
	} else {
		for idx, req := range requests.GetRequests() {
			res, _ := s.Get(ctx, req)
			responses[idx] = res
		}
	}

	return &pbv.BatchGetResponse{Responses: responses}, nil
}

func (s *server) BatchSet(ctx context.Context, requests *pbv.BatchSetRequest) (*pbv.BatchSetResponse, error) {
	responses := make([]*pbv.SetResponse, len(requests.GetRequests()))

	if s.config.ParallelBatchProcessing {
		wg := &sync.WaitGroup{}
		for i, r := range requests.GetRequests() {
			wg.Add(1)
			go func(idx int, req *pbv.SetRequest) {
				defer wg.Done()
				// verify MVCC
				if !s.checkMVCC(ctx, req.GetKey(), req.GetVersion()) {
					responses[idx] = nil
					return
				}
				// prepare request
				req = s.prepareSetRequest(req)
				s.setRequestCh <- req
				responses[idx] = &pbv.SetResponse{Txid: req.TxId}
			}(i, r)
		}
		wg.Wait()
	} else {
		for idx, req := range requests.GetRequests() {
			res, _ := s.Set(ctx, req)
			responses[idx] = res
		}
	}

	return &pbv.BatchSetResponse{Responses: responses}, nil
}
