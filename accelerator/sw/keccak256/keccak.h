#ifdef __cplusplus
extern "C" {
#endif
    void init_kernel(char* xclbin, int num);    
    void run_kernel(int num, int size);
    int send_data(unsigned char* data, int* sizes, int num);
    unsigned char* get_results(int num);
#ifdef __cplusplus
}
#endif
