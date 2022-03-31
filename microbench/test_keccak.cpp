#include "keccak.hpp"

class ArgParser
{
public:
    ArgParser(int &argc, const char **argv)
    {
        for (int i = 1; i < argc; ++i)
            mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string &value) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end())
        {
            value = *itr;
            return true;
        }
        return false;
    }

private:
    std::vector<std::string> mTokens;
};

int main(int argc, const char *argv[])
{
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    std::string req_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path))
    {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    if (!parser.getCmdOption("-reqpath", req_path))
    {
        std::cout << "ERROR:requests path is not set!\n";
        return 1;
    }

    FILE *fin = fopen(req_path.c_str(), "rt");
    if (!fin)
    {
        std::cout << "ERROR: cannot open requests file!\n";
        return 1;
    }

    FILE *fout = fopen("output.txt", "wt");
    if (!fout)
    {
        std::cout << "ERROR: cannot open output file!\n";
        return 1;
    }

    KernelQueue kq = init(xclbin_path, BATCH_SIZE);

    for (int k = 0; k < 100; k++)
    {
        std::cout << "New batch ...\n";

        int size = read_data(fin, BATCH_SIZE);

        run(kq, BATCH_SIZE, size);

        write_data(fout, BATCH_SIZE);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}
