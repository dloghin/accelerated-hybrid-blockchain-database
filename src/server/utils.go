package server

import (
	"bufio"
	"crypto/ecdsa"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
)

// https://github.com/ethereum/go-ethereum/blob/master/crypto/crypto.go
// readASCII reads into 'buf', stopping when the buffer is full or
// when a non-printable control character is encountered.
func readASCII(buf []byte, r *bufio.Reader) (n int, err error) {
	for ; n < len(buf); n++ {
		buf[n], err = r.ReadByte()
		switch {
		case err == io.EOF || buf[n] < '!':
			return n, nil
		case err != nil:
			return n, err
		}
	}
	return n, nil
}

// https://github.com/ethereum/go-ethereum/blob/master/crypto/crypto.go
// checkKeyFileEnd skips over additional newlines at the end of a key file.
func checkKeyFileEnd(r *bufio.Reader) error {
	for i := 0; ; i++ {
		b, err := r.ReadByte()
		switch {
		case err == io.EOF:
			return nil
		case err != nil:
			return err
		case b != '\n' && b != '\r':
			return fmt.Errorf("invalid character %q at end of key file", b)
		case i >= 2:
			return errors.New("key file too long, want 64 hex characters")
		}
	}
}

// https://github.com/ethereum/go-ethereum/blob/master/crypto/crypto.go
// Loads a secp256k1 public key from the given file.
func LoadECDSAPub(file string) (*ecdsa.PublicKey, error) {
	fd, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer fd.Close()

	r := bufio.NewReader(fd)
	buf := make([]byte, 130)
	_, err = readASCII(buf, r)
	if err != nil {
		return nil, err
	}
	if err := checkKeyFileEnd(r); err != nil {
		return nil, err
	}
	decoded, err := hex.DecodeString(string(buf))
	if err != nil {
		return nil, err
	}
	return crypto.UnmarshalPubkey(decoded)
}

func VerifySignature(publicKeyStr, signatureStr, payload string) error {
	publicKey, err := hex.DecodeString(publicKeyStr)
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return err
	}
	signature, err := hex.DecodeString(signatureStr)
	if err != nil {
		fmt.Printf("Error in decode hex %v\n", err)
		return err
	}
	/*
		// for debugging only
		if len(signature) != 64 {
			fmt.Println("Error len signature != 64")
			return errors.New("Invalid Signature")
		}
	*/
	digest := crypto.Keccak256([]byte(payload))
	/*
		// for debugging only
		if len(digest) != 32 {
			fmt.Println("Error len digest != 32")
			return errors.New("Invalid Digest")
		}
	*/
	if !secp256k1.VerifySignature(publicKey, digest, signature) {
		return errors.New("Invalid Signature")
	}
	return nil
}
