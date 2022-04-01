/*
 * Copyright 2020 - 2022 Dumitrel Loghin & NUS DB Lab
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
	"log"
	"net"
	"os"
	"os/signal"
	"runtime/pprof"
	"strings"

	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"

	pbv "hbdb/proto/hbdb"
	"hbdb/src/dbconn"
	"hbdb/src/kafkarole"
	"hbdb/src/server"
)

var (
	signature               = kingpin.Flag("signature", "server signature").Required().String()
	blockSize               = kingpin.Flag("blk-size", "block size").Default("100").Int()
	parties                 = kingpin.Flag("parties", "party1,party2,...").Required().String()
	addr                    = kingpin.Flag("addr", "server address").Required().String()
	kafkaAddr               = kingpin.Flag("kafka-addr", "kafka server address").Required().String()
	kafkaGroup              = kingpin.Flag("kafka-group", "kafka group id").Required().String()
	kafkaTopic              = kingpin.Flag("kafka-topic", "kafka topic").Required().String()
	redisAddr               = kingpin.Flag("redis-addr", "redis server address").Required().String()
	redisDb                 = kingpin.Flag("redis-db", "redis db number").Required().Int()
	redisPwd                = kingpin.Flag("redis-pwd", "redis password").String()
	ledgerPath              = kingpin.Flag("ledger-path", "ledger path").Required().String()
	parallelBatchProcessing = kingpin.Flag("parallel-batch-processing", "parallel request batch processing").Default("false").Bool()
	xclbinPath              = kingpin.Flag("xclbin-path", "xclbin path").Required().String()
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	f, err := os.Create("/tmp/peer-cpu.pprof")
	if err != nil {
		log.Fatal(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	kingpin.Parse()

	r, err := dbconn.NewRedisConn(*redisAddr, *redisPwd, *redisDb)
	check(err)

	c, err := kafkarole.NewConsumer(*kafkaAddr, *kafkaGroup, []string{*kafkaTopic})
	check(err)
	p, err := kafkarole.NewProducer(*kafkaAddr, *kafkaTopic)
	check(err)

	pm := make(map[string]struct{})
	pm[*signature] = struct{}{}

	ps := strings.Split(*parties, ",")
	for _, s := range ps {
		pm[s] = struct{}{}
	}

	s := grpc.NewServer()
	svr := server.NewServer(r, c, p, *ledgerPath, *xclbinPath, &server.Config{
		Signature:               *signature,
		Topic:                   *kafkaTopic,
		Parties:                 pm,
		BlockSize:               *blockSize,
		ParallelBatchProcessing: *parallelBatchProcessing,
	})
	pbv.RegisterNodeServer(s, svr)
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		panic(err)
	}

	go func() {
		log.Printf("Serving gRPC: %s", *addr)
		s.Serve(lis)
	}()

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt, os.Kill)
	sig := <-ch
	log.Printf("Received signal %v, quiting gracefully", sig)
}
