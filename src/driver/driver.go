package driver

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
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

	fmt.Println("***")
	fmt.Printf("Public: %v\n", pubkey)
	fmt.Printf("Private: %v\n", privkey)

	return pubkey, privkey
}

// taken from https://github.com/ethereum/go-ethereum/blob/master/crypto/secp256k1/secp256_test.go
func generateKeyPair() (pubkey, privkey []byte) {
	key, err := ecdsa.GenerateKey(secp256k1.S256(), rand.Reader)
	if err != nil {
		panic(err)
	}
	pubkey = elliptic.Marshal(secp256k1.S256(), key.X, key.Y)

	privkey = make([]byte, 32)
	blob := key.D.Bytes()
	copy(privkey[32-len(blob):], blob)

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
		fmt.Printf("Error in Driver signature Get %v\n", err)
		return "", -1, err
	}

	/*
		TODO - remove
		pubkey, err := hex.DecodeString(hex.EncodeToString(d.pubkey))
		if err != nil {
			fmt.Printf("Error in decode hex %v\n", err)
			return "", -1, err
		}
		sig, err := hex.DecodeString(hex.EncodeToString(signature))
		if err != nil {
			fmt.Printf("Error in decode hex %v\n", err)
			return "", -1, err
		}

		pubkeystr := hex.EncodeToString(d.pubkey)
		if len(pubkeystr) == 0 {
			fmt.Println("Empty public key")
		}

		if !secp256k1.VerifySignature(pubkey, dig, sig[:64]) {
			fmt.Println("Verify False Get")
		}
	*/

	res, err := d.dbCli.Get(ctx, &pbv.GetRequest{
		Pubkey:    d.pubkeystr,
		Signature: hex.EncodeToString(signature),
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
		fmt.Printf("Error in Driver signature Set %v\n", err)
		return err
	}
	if _, err := d.dbCli.Set(ctx, &pbv.SetRequest{
		Pubkey:    d.pubkeystr,
		Signature: hex.EncodeToString(signature),
		Key:       key,
		Value:     value,
		Version:   version,
	}); err != nil {
		return err
	}
	return nil
}

func (d *Driver) Close() error {
	return d.cc.Close()
}
