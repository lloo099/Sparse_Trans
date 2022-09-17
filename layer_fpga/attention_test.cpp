#include "layer.hpp"
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
#include "parameters.hpp"
// #include "tb_func.h"

void read_input_test(float *in, float *in_tem)
{

    for (int it = 0; it < CFG::dmodel / 128; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / 128; ++jt)
        {
            for (int kt = 0; kt < CFG::seqlen / 128; ++kt)
            {
                for (int k = 0; k < 128; ++k)
                {
                    for (int i = 0; i < 128; ++i)
                    {
                        float A_val = in[kt * 128 + k + (it * 128 + i) * CFG::dmodel];

                        // in_1.write(A_val);
                        // in_2.write(A_val);
                        // in_3.write(A_val);
                        in_tem[(it * 128 + i) * CFG::seqlen + kt * 128 + k] = in[(kt * 128 + k) * CFG::dmodel + it * 128 + i];
                    }
                }
            }
        }
    }
}

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

void printmat(float *A, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << float(A[i * N + j]) << ',';
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
const bool check(T *A, T *B, const int M, const int N)
{
    for (int i = 0; i < M * N; i++)
    {
        if (A[i] - B[i] > 0.005)
        {
            std::cout << "A: " << A[i] << "  B :" << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    freopen("../../../../csim_results_log.txt", "w", stdout);

    /*attention parameters*/
    float *hidden_states = new float[CFG::seqlen * CFG::dmodel];
    float *hidden_states_T = new float[CFG::seqlen * CFG::dmodel];

    float *query_weight_t = new float[CFG::dmodel * CFG::dmodel];
    float *key_weight_t = new float[CFG::dmodel * CFG::dmodel];
    float *value_weight_t = new float[CFG::dmodel * CFG::dmodel];
    float *query_bias = new float[CFG::dmodel];
    float *key_bias = new float[CFG::dmodel];
    float *value_bias = new float[CFG::dmodel];
    float *dense_weight_t = new float[CFG::dmodel * CFG::dmodel];
    float *dense_bias = new float[CFG::dmodel];
    float *norm_weight = new float[CFG::dmodel];
    float *norm_bias = new float[CFG::dmodel];

    /*ffn parameters*/
    float *ffn_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
    float *ffn_dense_bias = new float[CFG::ffdim];
    float *ffn_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
    float *ffn_dense2_bias = new float[CFG::dmodel];
    float *ffn_norm_weight = new float[CFG::dmodel];
    float *ffn_norm_bias = new float[CFG::dmodel];
    load_from_txt<float, CFG::seqlen * CFG::dmodel>(hidden_states, "embedding_in_hls.txt");

    for (int i = 0; i < CFG::dmodel; ++i)
    {
        for (int j = 0; j < CFG::seqlen; ++j)
        {
            hidden_states_T[i * CFG::seqlen + j] = hidden_states[j * CFG::dmodel + i];
            // printf("%f ", hidden_states[j * CFG::dmodel + i]);
        }
        // printf("\n");
    }

    // printf("\n");
    float *hidden_states_T_fake = new float[CFG::seqlen * CFG::dmodel];
    read_input_test(hidden_states, hidden_states_T_fake);

    std::cout << "transpose test: " << (check(hidden_states_T_fake, hidden_states_T, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    load_from_txt<float, CFG::dmodel * CFG::dmodel>(query_weight_t, "encoder1_query_weight_hls.txt");
    load_from_txt<float, CFG::dmodel * CFG::dmodel>(key_weight_t, "encoder1_key_weight_hls.txt");
    load_from_txt<float, CFG::dmodel * CFG::dmodel>(value_weight_t, "encoder1_value_weight_hls.txt");
    load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense_weight_t, "encoder1_dense1_weight_hls.txt");
    load_from_txt<float, CFG::dmodel>(norm_weight, "encoder1_lnorm_weight_hls.txt");

    load_from_txt<float, CFG::dmodel>(query_bias, "encoder1_query_bias_hls.txt");
    load_from_txt<float, CFG::dmodel>(key_bias, "encoder1_key_bias_hls.txt");
    load_from_txt<float, CFG::dmodel>(value_bias, "encoder1_value_bias_hls.txt");
    load_from_txt<float, CFG::dmodel>(dense_bias, "encoder1_dense1_bias_hls.txt");
    load_from_txt<float, CFG::dmodel>(norm_bias, "encoder1_lnorm_bias_hls.txt");

    load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn_dense_weight_t, "encoder1_ffn1_weight_hls.txt");
    load_from_txt<float, CFG::ffdim>(ffn_dense_bias, "encoder1_ffn1_bias_hls.txt");
    load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn_dense2_weight_t, "encoder1_ffn2_weight_hls.txt");
    load_from_txt<float, CFG::dmodel>(ffn_dense2_bias, "encoder1_ffn2_bias_hls.txt");
    load_from_txt<float, CFG::dmodel>(ffn_norm_weight, "encoder1_ffn2_lnweight_hls.txt");
    load_from_txt<float, CFG::dmodel>(ffn_norm_bias, "encoder1_ffn2_lnbias_hls.txt");
    // // printmat(query_weight_t, CFG::dmodel, CFG::dmodel);
    // // printf("dense weight test = : \n");
    // // printmat(dense_weight_t, CFG::dmodel, CFG::dmodel);
    auto encoder1_attention_out = new float[CFG::seqlen * CFG::dmodel];
    // auto attention_pyout = new float[CFG::seqlen * CFG::dmodel];
    // auto attention_pyout_T = new float[CFG::seqlen * CFG::dmodel];
    // load_from_txt<float, CFG::seqlen * CFG::dmodel>(attention_pyout, "encoder1_attention_out_hls.txt");

    // for (int i = 0; i < CFG::dmodel; ++i)
    // {
    //     for (int j = 0; j < CFG::seqlen; ++j)
    //     {
    //         attention_pyout_T[i * CFG::seqlen + j] = attention_pyout[j * CFG::dmodel + i];
    //     }
    // }
    // /**
    //  * Linear test
    //  */
    // auto query_in_gt = new float[CFG::seqlen * CFG::dmodel];
    // auto key_in_gt = new float[CFG::seqlen * CFG::dmodel];
    // auto value_in_gt = new float[CFG::seqlen * CFG::dmodel];
    // load_from_txt<float, CFG::seqlen * CFG::dmodel>(query_in_gt, "encoder1_q_layer_out_hls.txt");
    // load_from_txt<float, CFG::seqlen * CFG::dmodel>(key_in_gt, "encoder1_k_layer_out_hls.txt");
    // load_from_txt<float, CFG::seqlen * CFG::dmodel>(value_in_gt, "encoder1_v_layer_out_hls.txt");

    // auto query_in = new float[CFG::seqlen * CFG::dmodel];
    // auto key_in = new float[CFG::seqlen * CFG::dmodel];
    // auto value_in = new float[CFG::seqlen * CFG::dmodel];

    // float *query_weight_t_T = new float[CFG::dmodel * CFG::dmodel];
    // float *key_weight_t_T = new float[CFG::dmodel * CFG::dmodel];
    // float *value_weight_t_T = new float[CFG::dmodel * CFG::dmodel];
    // float *dense_weight_t_T = new float[CFG::dmodel * CFG::dmodel];
    // float *k_T = new float[CFG::seqlen * CFG::dmodel];

    // for (int i = 0; i < CFG::dmodel; ++i)
    // {
    //     for (int j = 0; j < CFG::dmodel; ++j)
    //     {
    //         query_weight_t_T[i * CFG::dmodel + j] = query_weight_t[j * CFG::dmodel + i];
    //         key_weight_t_T[i * CFG::dmodel + j] = key_weight_t[j * CFG::dmodel + i];
    //         value_weight_t_T[i * CFG::dmodel + j] = value_weight_t[j * CFG::dmodel + i];
    //         dense_weight_t_T[i * CFG::dmodel + j] = dense_weight_t[j * CFG::dmodel + i];
    //         // printf("%f ", hidden_states_T[i * CFG::seqlen + j]);
    //     }
    //     // printf("\n");
    // }

    // // qkv(hidden_states, query_weight_t_T, query_bias, key_weight_t_T, key_bias, value_weight_t_T, value_bias,
    // //     query_in, key_in, value_in);

    // /**
    //  * Attention Test
    //  */
    // attention_layer(hidden_states, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
    //                 dense_weight_t, dense_bias, norm_weight, norm_bias, encoder1_attention_out);
    // printf("input test = : \n");
    // for (int i = 0; i < CFG::seqlen; ++i)
    // {
    //     for (int j = 0; j < CFG::dmodel; ++j)
    //     {

    //         printf("%f ", hidden_states[i * CFG::dmodel + j]);
    //     }
    //     printf("\n");
    // }
    // printf("query weight test = : \n");
    // printmat(query_weight_t, CFG::dmodel, CFG::dmodel);
    // attention_kernel(hidden_states, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
    //                  dense_weight_t, dense_bias, norm_weight, norm_bias, encoder1_attention_out);

    // // printf("attention test = : \n");
    // // printmat(encoder1_attention_out, CFG::seqlen, CFG::dmodel);
    // //  std::cout << "attention_out: " << (check(attention_pyout, encoder1_attention_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    // /**
    //  * ffn Test
    //  */
    // float *ffn_dense_weight_t_T = new float[CFG::dmodel * CFG::ffdim];
    // float *ffn_dense_out_gt = new float[CFG::seqlen * CFG::dmodel];
    // float *ffn_dense_out = new float[CFG::seqlen * CFG::dmodel];
    // float *ffn_dense2_weight_t_T = new float[CFG::ffdim * CFG::dmodel];
    float *encoder_layer1_gt = new float[CFG::seqlen * CFG::dmodel];
    // for (int i = 0; i < CFG::dmodel; ++i)
    // {
    //     for (int j = 0; j < CFG::ffdim; ++j)
    //     {
    //         ffn_dense_weight_t_T[i * CFG::ffdim + j] = ffn_dense_weight_t[j * CFG::dmodel + i];
    //     }
    // }

    // for (int i = 0; i < CFG::ffdim; ++i)
    // {
    //     for (int j = 0; j < CFG::dmodel; ++j)
    //     {
    //         ffn_dense2_weight_t_T[i * CFG::dmodel + j] = ffn_dense2_weight_t[j * CFG::ffdim + i];
    //     }
    // }
    // FFN(attention_pyout, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t, ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, ffn_dense_out_gt);
    // // linear_sw_test(attention_pyout, ffn_dense_weight_t_T, ffn_dense_bias, ffn_dense_out, CFG::seqlen, CFG::ffdim, CFG::dmodel);
    // printf("linear test = : \n");
    // printmat(ffn_dense_out_gt, CFG::seqlen, CFG::dmodel);

    // // std::cout << "query_out: " << (check(query_in_gt, query_in, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    // // std::cout << "key_out:   " << (check(key_in_gt, key_in, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    // // std::cout << "value_out: " << (check(value_in_gt, value_in, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    // // FFN_kernel(attention_pyout_T, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t, ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, ffn_dense_out);
    // // printf("hw linear test = : \n");
    // // printmat(ffn_dense_out, CFG::seqlen, CFG::dmodel);

    load_from_txt<float, CFG::seqlen * CFG::dmodel>(encoder_layer1_gt, "encoder1_ffn2_ln_layer_out_hls.txt");
    // // std::cout << "ffn out: " << (check(ffn_dense_out_gt, ffn_dense_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    float *encoder_dense_out = new float[CFG::seqlen * CFG::dmodel];
    // encoder(hidden_states, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
    //         dense_weight_t, dense_bias, norm_weight, norm_bias, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t,
    //         ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, encoder_dense_out);

    encoder_kernel(hidden_states, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
                   dense_weight_t, dense_bias, norm_weight, norm_bias, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t,
                   ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, encoder_dense_out);
    printf("encoder1 test = : \n");
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        for (int j = 0; j < CFG::dmodel; ++j)
        {

            printf("%f ", encoder_dense_out[i * CFG::dmodel + j]);
        }
        printf("\n");
    }
    std::cout << "encoder 1 out: " << (check(encoder_layer1_gt, encoder_dense_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
}
