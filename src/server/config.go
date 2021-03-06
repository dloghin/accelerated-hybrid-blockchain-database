package server

type Config struct {
	Signature               string
	Topic                   string
	Parties                 map[string]struct{}
	BlockSize               int
	ParallelBatchProcessing bool
}
