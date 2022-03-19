package driver

import (
	"context"

	"google.golang.org/grpc"

	pbv "hbdb/proto/hbdb"
)

type Driver struct {
	cc    *grpc.ClientConn
	dbCli pbv.NodeClient
}

func Open(serverAddr string) (*Driver, error) {
	cc, err := grpc.Dial(serverAddr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	dbCli := pbv.NewNodeClient(cc)

	return &Driver{
		cc:    cc,
		dbCli: dbCli,
	}, nil
}

func (d *Driver) Get(ctx context.Context, key string) (string, int64, error) {
	res, err := d.dbCli.Get(ctx, &pbv.GetRequest{
		Key: key,
	})
	if err != nil {
		return "", -1, err
	}
	return res.GetValue(), res.GetVersion(), nil
}

func (d *Driver) Set(ctx context.Context, key, value string, version int64) error {
	if _, err := d.dbCli.Set(ctx, &pbv.SetRequest{
		Key:     key,
		Value:   value,
		Version: version,
	}); err != nil {
		return err
	}
	return nil
}

func (d *Driver) BatchGet(ctx context.Context, size int, keys []string) ([]string, []int64, error) {
	requests := make([]*pbv.GetRequest, size)
	for idx := 0; idx < size; idx++ {
		requests[idx] = &pbv.GetRequest{
			Key: keys[idx],
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
		requests[idx] = &pbv.SetRequest{
			Key:     keys[idx],
			Value:   values[idx],
			Version: versions[idx],
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
