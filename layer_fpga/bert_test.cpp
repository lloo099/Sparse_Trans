#include "bert_fpga.hpp"
#include "config.hpp"
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
using namespace std;
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <map>
#include "hls_stream.h"
// #include "tb_func.h"

void linear_sw_test(float *A, float *B, float *bias, float *out, const int OH, const int OW, const int K)
{
    // B[K][OW] * A[OH][K]
    for (int i = 0; i < OH; i++)
    {
        for (int j = 0; j < OW; j++)
        {
            // Initialize accumulator
            out[i * OW + j] = bias[j];
            for (int k = 0; k < K; k++)
            {
                out[i * OW + j] += A[i * K + k] * B[k * OW + j];
            }
        }
    }
}

void sum3(float *A, float *B, float *C, float *out, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out[i * N + j] = A[i * N + j] + B[i * N + j] + C[i * N + j];
        }
    }
}

template <class T, size_t SIZE>
void load_from_txt(T *w, const char *fname)
{

    std::string full_path = std::string("/home/enai/Downloads/trans-fat/src/baseline/layer/parameters/") + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail())
    {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while (std::getline(iss, token, ','))
        {
            std::istringstream(token) >> w[i];
            i++;
        }

        if (SIZE != i)
        {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

void printmat_c(float *A, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << float(A[i * N + j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void genmat(T *A, const int M, const int N, const int mod)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (i * N + j) % mod;
        }
    }
}

template <typename T>
void posgen(T *A, const int M)
{
    float k = 0;
    for (int i = 0; i < M; i++)
    {

        A[i] = k;
        k = k + 1;
    }
}

template <typename T>
const bool check(T *A, T *B, const int M, const int N)
{
    for (int i = 0; i < M * N; i++)
    {
        if (A[i] - B[i] > 0.01)
        {
            std::cout << "A: " << A[i] << "  B :" << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    freopen("../../../../csim_bert_results_log.txt", "w", stdout);

    /**
     * Embedding test
     */

    float *input_ids = new float[CFG::seqlen];
    float *token_ids = new float[CFG::seqlen];
    float *pos_ids = new float[CFG::seqlen];
    float *dense_out = new float[CFG::dmodel];
    float *out_gt = new float[CFG::dmodel];

    load_from_txt<float, CFG::seqlen>(input_ids, "encoded_input_hls.txt");
    load_from_txt<float, CFG::seqlen>(token_ids, "encoded_token_hls.txt");
    load_from_txt<float, CFG::dmodel>(out_gt, "pooler_out_gt_hls.txt");

    posgen(pos_ids, CFG::seqlen);

    bert_fpga1(input_ids, token_ids, pos_ids, dense_out);
    printf("dense test = : \n");
    printmat_c(dense_out, 1, CFG::dmodel);

    std::cout << "BERT out: " << (check(out_gt, dense_out, 1, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
}
