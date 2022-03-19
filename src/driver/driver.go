package driver

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"fmt"

	"encoding/hex"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"google.golang.org/grpc"

	pbv "hbdb/proto/hbdb"
)

type Driver struct {
	pubkey    []byte
	pubkeystr string
	privkey   []byte
	cc        *grpc.ClientConn
	dbCli     pbv.NodeClient
}

// taken from https://github.com/ethereum/go-ethereum/blob/master/crypto/secp256k1/secp256_test.go
func DecodeKeyPair(key *ecdsa.PrivateKey) (pubkey, privkey []byte) {
	pubkey = elliptic.Marshal(secp256k1.S256(), key.X, key.Y)

	privkey = make([]byte, 32)
	blob := key.D.Bytes()
	copy(privkey[32-len(blob):], blob)

	//fmt.Println("***")
	//fmt.Printf("Public: %v\n", pubkey)
	//fmt.Printf("Private: %v\n", privkey)

	return pubkey, privkey
}

func Open(serverAddr string, pvk *ecdsa.PrivateKey) (*Driver, error) {
	cc, err := grpc.Dial(serverAddr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	dbCli := pbv.NewNodeClient(cc)

	pubkey, privkey := DecodeKeyPair(pvk)

	return &Driver{
		pubkey:    pubkey,
		pubkeystr: hex.EncodeToString(pubkey),
		privkey:   privkey,
		cc:        cc,
		dbCli:     dbCli,
	}, nil
}

func (d *Driver) Get(ctx context.Context, key string) (string, int64, error) {
	dig := crypto.Keccak256([]byte(key))
	signature, err := secp256k1.Sign(dig, d.privkey)
	if err != nil {
		fmt.Printf("[Driver] Error in Sign for Get: %v\n", err)
		return "", -1, err
	}

	res, err := d.dbCli.Get(ctx, &pbv.GetRequest{
		Pubkey:    d.pubkeystr,
		Signature: hex.EncodeToString(signature[:64]),
		Key:       key,
	})
	if err != nil {
		return "", -1, err
	}
	return res.GetValue(), res.GetVersion(), nil
}

func (d *Driver) Set(ctx context.Context, key, value string, version int64) error {
	data := fmt.Sprintf("%s%s%d", key, value, version)
	dig := crypto.Keccak256([]byte(data))
	signature, err := secp256k1.Sign(dig, d.privkey)
	if err != nil {
		fmt.Printf("[Driver] Error in Driver Sign for Set: %v\n", err)
		return err
	}
	if _, err := d.dbCli.Set(ctx, &pbv.SetRequest{
		Pubkey:    d.pubkeystr,
		Signature: hex.EncodeToString(signature[:64]),
		Key:       key,
		Value:     value,
		Version:   version,
	}); err != nil {
		return err
	}
	return nil
}

func (d *Driver) BatchGet(ctx context.Context, size int, keys []string) ([]string, []int64, error) {
	requests := make([]*pbv.GetRequest, size)
	for idx := 0; idx < size; idx++ {
		dig := crypto.Keccak256([]byte(keys[idx]))
		signature, err := secp256k1.Sign(dig, d.privkey)
		if err != nil {
			fmt.Printf("[Driver] Error in Sign for BatchGet: %v\n", err)
			return nil, nil, err
		}
		requests[idx] = &pbv.GetRequest{
			Pubkey:    d.pubkeystr,
			Signature: hex.EncodeToString(signature[:64]),
			Key:       keys[idx],
		}
	}

	res, err := d.dbCli.BatchGet(ctx, &pbv.BatchGetRequest{
		Requests: requests,
	})
	if err != nil {
		return nil, nil, err
	}
	vals := make([]string, size)
	vers := make([]int64, size)
	for idx := 0; idx < size; idx++ {
		vals[idx] = res.GetResponses()[idx].Value
		vers[idx] = res.GetResponses()[idx].Version
	}
	return vals, vers, nil
}

func (d *Driver) BatchSet(ctx context.Context, size int, keys, values []string, versions []int64) error {
	requests := make([]*pbv.SetRequest, size)
	for idx := 0; idx < size; idx++ {
		data := fmt.Sprintf("%s%s%d", keys[idx], values[idx], versions[idx])
		dig := crypto.Keccak256([]byte(data))
		signature, err := secp256k1.Sign(dig, d.privkey)
		if err != nil {
			fmt.Printf("[Driver] Error in Sign for BatchSet: %v\n", err)
			return err
		}
		requests[idx] = &pbv.SetRequest{
			Pubkey:    d.pubkeystr,
			Signature: hex.EncodeToString(signature[:64]),
			Key:       keys[idx],
			Value:     values[idx],
			Version:   versions[idx],
		}
	}

	if _, err := d.dbCli.BatchSet(ctx, &pbv.BatchSetRequest{
		Requests: requests,
	}); err != nil {
		return err
	}

	return nil
}

func (d *Driver) Close() error {
	return d.cc.Close()
}
