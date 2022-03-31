#include "keccak.hpp"
#include "keccak.h"

inline int tvdiff(struct timeval *tv0, struct timeval *tv1)
{
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}

template <typename T>
T *aligned_alloc(std::size_t num)
{
    void *ptr = NULL;

    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
        throw std::bad_alloc();

    return reinterpret_cast<T *>(ptr);
}

ap_uint<128> *msgLen;
ap_uint<256> *results;
ap_uint<64> *msg;

unsigned char *returnResults;

KernelQueue kq;

KernelQueue init(std::string xclbin_path, int num)
{
    KernelQueue kq;

    // alloc host memory
    msgLen = aligned_alloc<ap_uint<128>>(num);
    results = aligned_alloc<ap_uint<256>>(num);
    msg = aligned_alloc<ap_uint<64>>(num * MAX_DIM);

    // prepare kernel - platform related operations
    cl_int err = CL_SUCCESS;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    kq.queue = q;
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    cl::Kernel k(program, "keccak256_kernel", &err);
    kq.kernel = k;
    std::cout << "Kernel has been created" << std::endl;

    // memory
    cl_mem_ext_ptr_t mext_o[3];
    mext_o[0] = {2, msg, kq.kernel()};
    mext_o[1] = {3, msgLen, kq.kernel()};
    mext_o[2] = {4, results, kq.kernel()};

    // create device buffer and map dev buf to host buf
    kq.msg_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                            sizeof(ap_uint<64>) * BATCH_SIZE * MAX_DIM, &mext_o[0]);
    kq.msgLen_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               sizeof(ap_uint<128>) * BATCH_SIZE, &mext_o[1]);
    kq.digest_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                               sizeof(ap_uint<256>) * BATCH_SIZE, &mext_o[2]);

    kq.kernel.setArg(0, num);
    kq.kernel.setArg(1, 0);
    kq.kernel.setArg(2, kq.msg_buf);
    kq.kernel.setArg(3, kq.msgLen_buf);
    kq.kernel.setArg(4, kq.digest_buf);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Memory> ob_init;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_init.resize(0);
    ob_init.push_back(kq.msg_buf);
    ob_init.push_back(kq.msgLen_buf);

    kq.queue.enqueueMigrateMemObjects(ob_init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    kq.queue.finish();

    returnResults = (unsigned char *)malloc(num * 32);

    return kq;
}

// read and prepare data arrays
// return msg array size (each element is 4 bytes)
int read_data(FILE *fin, int num)
{
    // read data
    char line[256]; 
    size_t len = 0;
    int offidx = 0;
    for (int i = 0; i < num; i++)
    {
        // if (getline(&line, &len, fin) == -1)
	if (!fgets(line, 256, fin))
	{
        	std::cout << "No more data.\n";
        	break;
        }

        // TODO
        // len = 16;
	len = strlen(line);
	
	int len1 = len;
	if (len % 8 != 0) {
		len = 8 * (len / 8 + 1);
	}
	// printf("Original %d, new %d\n", len1, len);
	
        results[i] = 0;
        msgLen[i].range(127, 64) = (uint64_t)0;
        msgLen[i].range(63, 0) = (uint64_t)len;
        msg[offidx] = 0;
        for (int j = 0; j < (int)len1; j++)
        {
            int idx2 = (j % 8) * 8;
            msg[offidx].range(idx2 + 7, idx2) = (unsigned)line[j];
            if (j % 8 == 7)
            {
                offidx++;
                msg[offidx] = 0;
            }
        }
	
	for (int j = (int)len1; j < (int)len; j++)
        {
            int idx2 = (j % 8) * 8;
            msg[offidx].range(idx2 + 7, idx2) = 0;
            if (j % 8 == 7)
            {
                offidx++;
                msg[offidx] = 0;
            }
        }
    }
    return offidx;
}

void run(KernelQueue kq, int num, int size)
{
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Memory> ob_init;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    kq.kernel.setArg(0, num);
    kq.kernel.setArg(1, size);
    kq.kernel.setArg(2, kq.msg_buf);
    kq.kernel.setArg(3, kq.msgLen_buf);
    kq.kernel.setArg(4, kq.digest_buf);

    ob_in.resize(0);
    ob_in.push_back(kq.msg_buf);
    ob_in.push_back(kq.msgLen_buf);
    ob_out.resize(0);
    ob_out.push_back(kq.digest_buf);

    events_write.resize(1);
    events_kernel.resize(1);
    events_read.resize(1);

    // struct timeval start_time, end_time;

    // launch kernel and calculate kernel execution time
    // std::cout << "kernel start------" << std::endl;
    // gettimeofday(&start_time, 0);

    kq.queue.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    kq.queue.enqueueTask(kq.kernel, &events_write, &events_kernel[0]);
    kq.queue.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    kq.queue.finish();

    // gettimeofday(&end_time, 0);
    // std::cout << "kernel end------" << std::endl;
    // std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;
}

// Write digest to file
void write_data(FILE *fout, int num)
{
    unsigned char res;
    for (int i = 0; i < num; i++)
    {
        fprintf(fout, "0x");
        for (int j = 0; j < 32; j++)
        {
            res = results[i].range(8 * j + 7, 8 * j);
            fprintf(fout, "%02x", res);
        }
        fprintf(fout, "\n");
    }
}

// C functions to be called from Go
void init_kernel(char *xcl, int num)
{
    std::string xclfile(xcl);
    kq = init(xclfile, num);
}

void run_kernel(int num, int size)
{
    run(kq, num, size);
}

int send_data(unsigned char *data, int *sizes, int num)
{
    int inpidx = 0;
    int offidx = 0;
    for (int i = 0; i < num; i++)
    {
        results[i] = 0;
        msgLen[i].range(127, 64) = (uint64_t)0;
        msgLen[i].range(63, 0) = (uint64_t)sizes[i];
        msg[offidx] = 0;
        bool next = true;
        for (int j = 0; j < (int)sizes[i]; j++)
        {
            next = true;
            int idx2 = (j % 8) * 8;
            msg[offidx].range(idx2 + 7, idx2) = data[inpidx];
            inpidx++;
            if (j % 8 == 7)
            {
                offidx++;
                msg[offidx] = 0;
                next = false;
            }
        }
	// printf("%d -> %d\n", sizes[i], offidx);
        if (next)
        {
            offidx++;
            msg[offidx] = 0;
        }
    }
    return offidx;
}

unsigned char *get_results(int num)
{
    for (int i = 0; i < num; i++)
    {
        // printf("0x");
        for (int j = 0; j < 32; j++)
        {
            returnResults[32 * i + j] = results[i].range(8 * j + 7, 8 * j);
            // printf("%x", returnResults[32 * i + j]);
        }
        // printf("\n");
    }
    return returnResults;
}
