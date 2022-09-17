#include <inttypes.h>
#include <iostream>
#include <cmath>
#include "config.hpp"
#include "hls_stream.h"
#include <algorithm>
#include "mem.hpp"

void matrix2d_transpose(float *in, float *out, const int A, const int B)
{
    for (int i = 0; i < A; ++i)
    {
        for (int j = 0; j < B; ++j)
        {
            out[i * B + j] = in[j * A + i];
        }
    }
}
void embedding_lut(float *in, float *out, float *index, const int A, const int B)
{
    // word_embedding[word][B] A is the  sequence length, but B is the dmodel

    for (int k = 0; k < A; k++)
    {
        for (int j = 0; j < B; j++)
        {
            int lut = index[k];
            out[j + k * B] = in[lut * B + j];
        }
    }
}

void linear_sw_base(float *A, float *B, float *bias, float *out, const int N, const int M, const int K)
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            // Initialize accumulator
            out[i * M + j] = bias[j];
            for (int k = 0; k < K; k++)
            {
                out[i * M + j] += A[i * K + k] * B[k * M + j];
            }
        }
    }
}

// void linear_sw_stream(hls::stream<float> &A_stream, hls::stream<float> &B_stream, hls::stream<float> &bias_stream, hls::stream<float> &out_stream, const int N, const int M, const int K)
// {

//     for (int i = 0; i < N; i++)
//     {
//         for (int j = 0; j < M; j++)
//         {
//             // Initialize accumulator
//             out[i * M + j] = bias[j];
//             for (int k = 0; k < K; k++)
//             {
//                 out[i * M + j] += A[i * K + k] * B[k * M + j];
//             }
//         }
//     }
// }

void linear_fused1(hls::stream<float> &A_stream, hls::stream<float> &B_stream, hls::stream<float> &bias_stream, hls::stream<float> &out_stream)
{
    // buffers for tile mmult
    float out_block[TILE_SIZE1][TILE_SIZE1];
    float B_line[TILE_SIZE1];
    float A_line[TILE_SIZE1];

#pragma HLS array_partition dim = 2 complete variable = out_block
// #pragma HLS array_partition dim = 1 complete variable = out_block
#pragma HLS array_partition dim = 1 complete variable = B_line

    for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt)
        {

            // initialize output with bias
            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1; ++j)
                {
                    out_block[i][j] = bias_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE1; ++kt)
            {

                for (int k = 0; k < TILE_SIZE1; ++k)
                {

                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE1; ++j)
                    {
                        B_line[j] = B_stream.read();
                    }
                    for (int i = 0; i < TILE_SIZE1; ++i)
                    {
                        A_line[i] = A_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE1; ++i)
                    {
#pragma HLS PIPELINE II = 1
                        float Ai = A_line[i];
                        for (int j = 0; j < TILE_SIZE1; ++j)
                        {
#pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }

            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1; ++j)
                {
                    out_stream.write(float(out_block[i][j]));
                }
            }
        }
    }
}

void linear_pooler(float *A, float *B, float *bias, float *out, const int OH, const int OW, const int K)
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
            out[i * OW + j] = tanh(out[i * OW + j]);
        }
    }
}

void softmax_base(float *input, float *output)
{
    double m, sum, constant;
    int j;

    for (int n = 0; n < CFG::nhead; n++)
    {
        for (int i = 0; i < CFG::seqlen; i++)
        {
            // find max elem
            m = input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen];
            for (j = 0; j < CFG::seqlen; ++j)
            {
                if (m < input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j])
                {
                    m = input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j];
                }
            }

            sum = 0.0;
            for (j = 0; j < CFG::seqlen; ++j)
            {
                sum += exp(input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j] - m);
            }

            constant = m + log(sum);
            for (j = 0; j < CFG::seqlen; ++j)
            {
                output[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j] = float(exp(input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j] - constant));
            }
        }
    }
}

void gelu_base(float *gelu_in, float *gelu_out, int rows, int cols)
{
    // float PI = 3.14159265358979323846;
    float PI = 3.1416;
    float A = 0.044715;
    float a = sqrt(2.0 / PI);
    // float k = 1.4142;
    // int constant = 14;
    // float coef_0 = -0.2888;
    // float coef_1 = -1.769;
    // float coef_2 = 1 / coef_0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float val = gelu_in[i * cols + j];
            float pow3 = val * val * val;
            float b = val + A * pow3;

            // int32_t sign = (val >= 0) ? 1 : -1;
            // int32_t val_abs = val * sign;
            // int32_t abs_int = std::min(val_abs, -1 * b_int);
            // int32_t intermediate = (abs_int + b_int);
            // int32_t y_int = sign * (intermediate * intermediate + c_int);
            // int32_t sigmoid_int = y_int / (1 << constant);

            val = 0.5 * val * (1 + tanh(a * b));

            gelu_out[i * cols + j] = val;
        }
    }
}

void softmax_nn(float *input, float *output)
{
    double m, sum, constant;
    int j;

    for (int n = 0; n < CFG::nhead; n++)
    {
        for (int i = 0; i < CFG::seqlen; i++)
        {
            // find max elem
            // m = input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen];
            // for (j = 0; j < CFG::seqlen; ++j)
            // {
            //     if (m < input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j])
            //     {
            //         m = input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j];
            //     }
            // }

            sum = 0.0;
            for (j = 0; j < CFG::seqlen; ++j)
            {
                sum += exp(input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j]);
            }

            for (j = 0; j < CFG::seqlen; ++j)
            {
                output[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j] = float(exp(input[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + j]) / sum);
            }
        }
    }
}

void scale(float *y)
{
    float divisor = std::sqrt(CFG::dhead);
    printf("attention scale test = : \n");
    for (int i = 0; i < CFG::nhead * CFG::dhead * CFG::seqlen; ++i)
    {
        y[i] /= divisor;
        printf("%f ", y[i]);
    }
}

void add_skip2(float *inout, float *skip_conn, const int32_t len)
{
    for (int i = 0; i < len; ++i)
    {
        inout[i] += skip_conn[i];
    }
}

void add_skip_stream(int it, int jt, float out_block[TILE_SIZE2][TILE_SIZE2], float skip_buff[TILE_SIZE2][TILE_SIZE2], float *out)
{

    for (int i = 0; i < TILE_SIZE2; ++i)
    {
        for (int j = 0; j < TILE_SIZE2; ++j)
        {
            out[(it * TILE_SIZE2 + i) * CFG::dmodel + jt * TILE_SIZE2 + j] = out_block[i][j] + skip_buff[i][j];
        }
    }
}

void mean2(float *act, float *out)
{
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        // #pragma HLS unroll

        float acc32 = 0;
        for (int j = 0; j < CFG::dmodel; ++j)
        {
            acc32 += act[i * CFG::dmodel + j];
        }
        out[i] = float(acc32 / CFG::dmodel);
    }
}

void sum2(float *in, float *out)
{
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        // #pragma HLS unroll

        out[i] = 0;
        for (int j = 0; j < CFG::dmodel; ++j)
        {
            out[i] += in[i * CFG::dmodel + j];
        }
    }
}

void sum3_base(float *A, float *B, float *C, float *out, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out[i * N + j] = A[i * N + j] + B[i * N + j] + C[i * N + j];
        }
    }
}

void diff2(float *y, float *act, float *means)
{
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        // #pragma HLS unroll

        for (int j = 0; j < CFG::dmodel; ++j)
        {
            y[i * CFG::dmodel + j] = act[i * CFG::dmodel + j] - means[i];
        }
    }
}

void div2(float *y, float *stdev)
{
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        // #pragma HLS unroll

        for (int j = 0; j < CFG::dmodel; ++j)
        {
            y[i * CFG::dmodel + j] /= stdev[i];
        }
    }
}

void layernorm_base(float *act, float *y, float *norm_weight, float *norm_bias)
{
    // float *means = new float[CFG::seqlen];
    // float *y_sq = new float[CFG::seqlen * CFG::dmodel];
    // float *var = new float[CFG::seqlen];
    // float *stdev = new float[CFG::seqlen];

    float means[CFG::seqlen];
    float y_sq[CFG::seqlen * CFG::dmodel];
    float var[CFG::seqlen];
    float stdev[CFG::seqlen];
    mean2(act, means);
    diff2(y, act, means);

    // square elements
    for (int i = 0; i < CFG::dmodel * CFG::seqlen; ++i)
    {
        y_sq[i] = float(float((y[i] * y[i])) / CFG::dmodel);
    }

    // compute var by summing y^2
    sum2(y_sq, var);

    // calculate constant for std computation
    float C = float(CFG::eps);

    // compute std
    for (int i = 0; i < CFG::seqlen; ++i)
    {
        stdev[i] = float(sqrt(float(var[i] + C)));
    }

    // perform the division on each element in y
    div2(y, stdev);

    // perform macs
    for (int i = 0; i < CFG::dmodel; ++i)
    {
        for (int j = 0; j < CFG::seqlen; ++j)
        {
            y[j * CFG::dmodel + i] = float((y[j * CFG::dmodel + i] * norm_weight[i] + norm_bias[i]));
        }
    }
}

void layernorm_base_t(float *act, float *y, float *norm_weight, float *norm_bias)
{
    // float *means = new float[CFG::seqlen];
    // float *y_sq = new float[CFG::seqlen * CFG::dmodel];
    // float *var = new float[CFG::seqlen];
    // float *stdev = new float[CFG::seqlen];

    float means[CFG::seqlen];
    float y_sq[CFG::dmodel * CFG::seqlen];
    float var[CFG::seqlen];
    float stdev[CFG::seqlen];
#pragma HLS array_partition dim = 1 complete variable = var
#pragma HLS array_partition dim = 1 complete variable = stdev
#pragma HLS array_partition dim = 1 complete variable = means
    // #pragma HLS array_partition dim = 2 complete variable = y_sq

    mean2(act, means);
    diff2(y, act, means);
    int constant = CFG::dmodel;

    for (int i = 0; i < CFG::dmodel * CFG::seqlen; ++i)
    {
        // #pragma HLS unroll
        y_sq[i] = float(float((y[i] * y[i])) / CFG::dmodel);
    }

    // square elements
    // int constant = CFG::dmodel;
    // for (int i = 0; i < CFG::dmodel; ++i)
    // {

    //     for (int j = 0; j < CFG::seqlen; ++j)
    //     {

    //         y_sq[i][j] = y[i * CFG::seqlen + j] * y[i * CFG::seqlen + j] / CFG::dmodel;
    //     }
    // }

    // compute var by summing y^2
    sum2(y_sq, var);
    //     for (int i = 0; i < CFG::seqlen; ++i)
    //     {
    // #pragma HLS unroll
    //         var[i] = 0;
    //         for (int j = 0; j < CFG::dmodel; ++j)
    //         {
    //             y_sq[j][i] = (y[i * constant + j] * y[i * constant + j]) / constant;
    //             var[i] += y_sq[j][i];
    //         }
    //     }

    // calculate constant for std computation
    float C = float(CFG::eps);

    // compute std
    for (int i = 0; i < CFG::seqlen; ++i)
    {
#pragma HLS unroll

        stdev[i] = float(sqrt(float(var[i] + C)));
    }

    // perform the division on each element in y
    div2(y, stdev);
    //     for (int i = 0; i < CFG::seqlen; ++i)
    //     {
    // #pragma HLS unroll

    //         for (int j = 0; j < CFG::dmodel; ++j)
    //         {
    //             // #pragma HLS unroll
    //             y[i * CFG::dmodel + j] /= stdev[i];
    //         }
    //     }
    // perform macs
    for (int i = 0; i < CFG::dmodel; ++i)
    {
        for (int j = 0; j < CFG::seqlen; ++j)
        {
#pragma HLS unroll

            y[j * CFG::dmodel + i] = float((y[j * CFG::dmodel + i] * norm_weight[i] + norm_bias[i]));
        }
    }
}

void embedding_out(float *word_embeddings, float *token_embeddings, float *pos_embeddings, float *input_ids, float *token_ids, float *pos_ids, float *embed_norm_weight, float *embed_norm_bias, float *dense_out)
{

    float word[CFG::seqlen * CFG::dmodel];
    float pos[CFG::seqlen * CFG::dmodel];
    float token[CFG::seqlen * CFG::dmodel];
    float dense_temp[CFG::seqlen * CFG::dmodel];
    embedding_lut(word_embeddings, word, input_ids, CFG::seqlen, CFG::dmodel);
    embedding_lut(token_embeddings, token, token_ids, CFG::seqlen, CFG::dmodel);
    embedding_lut(pos_embeddings, pos, pos_ids, CFG::seqlen, CFG::dmodel);
    sum3_base(word, token, pos, dense_temp, CFG::seqlen, CFG::dmodel);
    layernorm_base(dense_temp, dense_out, embed_norm_weight, embed_norm_bias);
}

/**
 *
 * What is transpose?
 * A[i][j] = A[i*COLS + j]
 * A_T[i][j] = A[j][i] = A[j*COLS+i]
 *
 * To obtain indexing into original array from transpose array:
 *  - for each dimension in the transpose array, find the corresponding dimension in the original
 *  - index into this transpose dimension in the place where the original dimension was
 *
 * This means:
 * if A[i][j][k] was transposed in a way that rotated the dimensions around the right, so it now looks like B[k][i][j],
 * the way you index B[i][j][k] is with A[j][k][i]. This makes the most sense if each dimension means something to you,
 * like nhead, seqlen, and dhead.
 *
 * Then, you can flatten you new reordered index by, for each index, multiplying it by the product of the dimensions
 * following it and adding that to your accumulated index.
 */

void attention_scores(float *query, float *key, float *out, const int seqlen, const int nhead, const int dhead)
{
    /*
     * query :   <seqlen, dmodel> -> <nhead, seqlen, dhead>
     * key:      <seqlen, dmodel> -> <nhead, dhead, seqlen>
     * out:      <nhead, seqlen, seqlen>
     *
     * query reshape: view as <seqlen, nhead, dhead> (means changing bounds and adding another dimension)
     * query_reshape[i][j][k] = query[i*nhead*dhead + j*dhead + k]
     * query transpose to get <nhead, seqlen, dhead> switches i and j.
     * query_transpose[i][j][k] = query_reshape[j][i][k] = query[j*nhead*dhead +i*dhead + k]
     *
     * key reshape: view as <seqlen, nhead, dhead>
     * key_reshape[i][j][k] = key[i*nhead*dhead + j*dhead + k]
     * key transpose to get to <nhead, dhead, seqlen>. go from (i,j,k) to (j,k,i)
     * key_transpose[i][j][k] = key_reshape[k][i][j] = key[k*nhead*dhead + i*dhead + j]
     *
     *
     * query_transpose[i1][i2][i3] = query[i2*nhead*dhead +i1*dhead + i3]
     * key_transpose[i1][i2][i3] = key[i3*nhead*dhead + i1*dhead + i2]
     * Summary: repeat nhead times, inter-loop is the matrix multiplication.
     */

    float divisor = std::sqrt(CFG::dhead);

    for (int n = 0; n < nhead; n++)
    {
        // compute matmul NHEAD times
        for (int i = 0; i < seqlen; i++)
        {
            for (int j = 0; j < seqlen; j++)
            {
                float accum = 0;
                for (int k = 0; k < dhead; k++)
                {
                    // accum += query[n,i,k] * key[n, k, j]
                    accum += query[i * nhead * dhead + n * dhead + k] * key[j * nhead * dhead + n * dhead + k];
                }
                // out[n,i,j] = accum
                accum /= divisor;
                out[n * seqlen * seqlen + i * seqlen + j] = accum;
                // printf("%f ", out[n * seqlen * seqlen + i * seqlen + j]);
            }
        }
    }
}

void attention_values(float *probs, float *value, float *attn_out, const int seqlen, const int nhead, const int dhead)
{

    /**
     * probs: <nhead, seqlen, seqlen>
     * value: <seqlen, dmodel> -> <nhead, seqlen, dhead> (same reshape/transpose as query)
     *
     * attn_out: <nhead, seqlen, dhead> -> <seqlen, dmodel> (how do you index to do this in one shot)
     * attn_out[i1][i2][i3] = attn_out[i1*seqlen*dhead + i2*dhead + i3]
     * att_out_transpose is <seqlen, nhead, dhead>
     * att_out_transpose[i1][i2][i3] = attn_out[i2][i1][i3] = attn_out[i2*nhead*dhead + i1*dhead + i3]
     *
     * value_transpose[i1][i2][i3] = value[i2*nhead*dhead +i1*dhead + i3]
     *
     */

    for (int n = 0; n < nhead; n++)
    {
        for (int i = 0; i < seqlen; i++)
        {
            for (int j = 0; j < dhead; j++)
            {
                float accum = 0;
                for (int k = 0; k < seqlen; k++)
                {
                    // attn_out[n][i][j] += probs[n][i][k] * value[n][k][j]
                    accum += probs[n * seqlen * seqlen + i * seqlen + k] * value[k * nhead * dhead + n * dhead + j];
                }
                // writes to attn_out to obtain <seqlen, dmodel> shape
                attn_out[i * nhead * dhead + n * dhead + j] = accum;
            }
        }
    }
}

void softmax_fused(float *input, float *output, int start_idx)
{

    int i;
    double m, sum, constant;

    // find max elem
    // m = input[0];
    // for (i = 0; i < CFG::seqlen; ++i)
    // {
    //     if (m < input[i])
    //     {
    //         m = input[i];
    //     }
    // }

    sum = 0.0;
    for (i = 0; i < CFG::seqlen; ++i)
    {
        sum += exp(input[i]);
    }

    for (i = 0; i < CFG::seqlen; ++i)
    {
        output[start_idx + i] = float(exp(input[i]) / sum);
        // output[start_idx + i] = float(input[i]);
    }
}

void attention_scores_fused(float *query, float *key, float *out)
{

    float divisor = std::sqrt(CFG::dhead);
    float rowbuff[CFG::seqlen];
    float query_row[CFG::dhead];
    float key_row[CFG::dhead];

#pragma HLS array_partition variable = query_row complete
#pragma HLS array_partition variable = key_row complete
#pragma HLS array_partition variable = rowbuff complete
    // #pragma HLS array_partition variable = divisor complete
    for (int n = 0; n < CFG::nhead; n++)
    {
#pragma HLS unroll

        // compute matmul NHEAD times
        for (int i = 0; i < CFG::seqlen; i++)
        {

            for (int k = 0; k < CFG::dhead; k++)
            {

                query_row[k] = query[i * CFG::nhead * CFG::dhead + n * CFG::dhead + k];
            }

            for (int j = 0; j < CFG::seqlen; j++)
            {

                for (int k = 0; k < CFG::dhead; k++)
                {

                    key_row[k] = key[j * CFG::nhead * CFG::dhead + n * CFG::dhead + k];
                }

                float accum = 0;
                for (int k = 0; k < CFG::dhead; k++)
                { // accum += query[n,i,k] * key[n, k, j]
#pragma HLS unroll
                    accum += query_row[k] * key_row[k];
                }
                // out[n,i,j] = accum
                rowbuff[j] = accum / divisor;
            }
            softmax_fused(rowbuff, out, n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen);
        }
    }
}

void attention_scores_fused_stream(hls::stream<float> &query, hls::stream<float> &key, float *out)
{

    float divisor = std::sqrt(CFG::dhead);
    float rowbuff[CFG::seqlen * CFG::seqlen];
    float query_buffer[CFG::seqlen][CFG::nhead * CFG::dhead];
    float key_buffer[CFG::seqlen][CFG::nhead * CFG::dhead];

#pragma HLS array_partition variable = query_buffer block factor = 12 dim = 2
#pragma HLS array_partition variable = key_buffer block factor = 12 dim = 2
// #pragma HLS array_partition variable = key_buffer complete
#pragma HLS array_partition variable = rowbuff block factor = 128 dim = 1
    // #pragma HLS array_partition variable = divisor complete
    for (int n = 0; n < CFG::nhead; n++)
    {
#pragma HLS unroll

        // compute matmul NHEAD times
        for (int i = 0; i < CFG::seqlen; i++)
        {

            for (int k = 0; k < CFG::dhead; k++)
            {

                // query_row[k] = query[i * CFG::nhead * CFG::dhead + n * CFG::dhead + k];
                query_buffer[i][n * CFG::dhead + k] = query.read();
            }

            for (int j = 0; j < CFG::seqlen; j++)
            {
#pragma HLS unroll

                for (int k = 0; k < CFG::dhead; k++)
                {

                    // key_row[k] = key[j * CFG::nhead * CFG::dhead + n * CFG::dhead + k];
                    key_buffer[j][n * CFG::dhead + k] = key.read();
                }

                float accum = 0;
                for (int k = 0; k < CFG::dhead; k++)
                { // accum += query[n,i,k] * key[n, k, j]
#pragma HLS unroll
                  // accum += query_row[k] * key_row[k];
                    accum += query_buffer[i][n * CFG::dhead + k] * key_buffer[j][n * CFG::dhead + k];
                }
                // out[n,i,j] = accum
                rowbuff[i * CFG::seqlen + j] = accum / divisor;
            }
            softmax_fused(rowbuff, out, n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen);
        }
    }
}

void linear_fused2_tem(float *A, hls::stream<float> &weight_stream, hls::stream<float> &bias_stream, float *out, hls::stream<float> &skip_stream)
{

    // buffers for tile mmult
    //     float out_block[CFG::dmodel / TILE_SIZE2][TILE_SIZE2][TILE_SIZE2];
    //     float skip_buff[CFG::dmodel / TILE_SIZE2][TILE_SIZE2][TILE_SIZE2];
    //     float B_line[CFG::dmodel / TILE_SIZE2][TILE_SIZE2];
    //     float Ai[CFG::seqlen][CFG::dmodel]

    // #pragma HLS array_partition dim = 3 complete variable = out_block
    // #pragma HLS array_partition dim = 1 complete variable = out_block
    // #pragma HLS array_partition dim = 3 complete variable = skip_buff
    // #pragma HLS array_partition dim = 1 complete variable = skip_buff
    // #pragma HLS array_partition dim = 2 complete variable = B_line
    // #pragma HLS array_partition dim = 1 complete variable = B_line
    // #pragma HLS array_partition dim = 2 type = cycle factor = 6 variable = Ai

    // #pragma HLS array_partition dim = 2 complete variable = out_block
    // #pragma HLS array_partition dim = 1 complete variable = B_line

    float out_block[TILE_SIZE2][TILE_SIZE2];
    float skip_buff[TILE_SIZE2][TILE_SIZE2];
    float B_line[TILE_SIZE2];

#pragma HLS array_partition dim = 2 complete variable = out_block
#pragma HLS array_partition dim = 1 complete variable = B_line
    for (int it = 0; it < CFG::seqlen / TILE_SIZE2; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE2; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE2; ++i)
            {
                for (int j = 0; j < TILE_SIZE2; ++j)
                {
#pragma HLS pipeline II = 1
                    out_block[i][j] = bias_stream.read();
                    skip_buff[i][j] = skip_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE2; ++kt)
            {
                for (int k = 0; k < TILE_SIZE2; ++k)
                {

                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE2; ++j)
                    {
                        B_line[j] = weight_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE2; ++i)
                    {
#pragma HLS PIPELINE II = 1
                        float Ai = A[(it * TILE_SIZE2 + i) * CFG::dmodel + kt * TILE_SIZE2 + k];
                        for (int j = 0; j < TILE_SIZE2; ++j)
                        {
#pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }

            for (int i = 0; i < TILE_SIZE2; ++i)
            {
                for (int j = 0; j < TILE_SIZE2; ++j)
                {
                    out[(it * TILE_SIZE2 + i) * CFG::dmodel + jt * TILE_SIZE2 + j] = out_block[i][j] + skip_buff[i][j];
                }
            }
        }
    }
}

void linear_fused2(float *A, hls::stream<float> &weight_stream, hls::stream<float> &bias_stream, float *out_T, hls::stream<float> &skip_stream)
{

    // buffers for tile mmult
    float out_block[TILE_SIZE2][TILE_SIZE2];
    float skip_buff[TILE_SIZE2][TILE_SIZE2];
    float B_line[TILE_SIZE2];

#pragma HLS array_partition dim = 2 complete variable = out_block
#pragma HLS array_partition dim = 1 complete variable = B_line

    for (int it = 0; it < CFG::seqlen / TILE_SIZE2; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE2; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE2; ++i)
            {
                for (int j = 0; j < TILE_SIZE2; ++j)
                {
#pragma HLS pipeline II = 1
                    out_block[i][j] = bias_stream.read();
                    skip_buff[i][j] = skip_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE2; ++kt)
            {
                for (int k = 0; k < TILE_SIZE2; ++k)
                {

                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE2; ++j)
                    {
                        B_line[j] = weight_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE2; ++i)
                    {
#pragma HLS PIPELINE II = 1
                        float Ai = A[(it * TILE_SIZE2 + i) * CFG::dmodel + kt * TILE_SIZE2 + k];
                        for (int j = 0; j < TILE_SIZE2; ++j)
                        {
#pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }

            add_skip_stream(it, jt, out_block, skip_buff, out_T);
        }
    }
}

void linear_fused4(hls::stream<float> &A_stream, hls::stream<float> &B_stream, hls::stream<float> &bias_stream, float *out, hls::stream<float> &skip_stream)
{

    // buffers for tile mmult
    float out_block[TILE_SIZE4][TILE_SIZE4_J];
    float skip_buff[TILE_SIZE4][TILE_SIZE4_J];
    float B_line[TILE_SIZE4_J];
    float A_T_line[TILE_SIZE4];

#pragma HLS array_partition dim = 2 complete variable = out_block
//#pragma HLS array_partition dim=1 factor=32 variable=out_block
#pragma HLS array_partition dim = 1 complete variable = B_line

    // TODO: Can we read skip_conn_T contiguously from memory to fill skip_buff?

    for (int it = 0; it < CFG::seqlen / TILE_SIZE4; ++it)
    {

        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE4_J; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE4; ++i)
            {
                for (int j = 0; j < TILE_SIZE4_J; ++j)
                {
                    out_block[i][j] = bias_stream.read();
                    skip_buff[i][j] = skip_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::ffdim / TILE_SIZE4; ++kt)
            {
                for (int k = 0; k < TILE_SIZE4; ++k)
                {
                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE4_J; ++j)
                    {
                        B_line[j] = B_stream.read();
                    }
                    for (int i = 0; i < TILE_SIZE4; ++i)
                    {
                        A_T_line[i] = A_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE4; ++i)
                    {
//#pragma HLS unroll factor=4
#pragma HLS PIPELINE
                        float Ai = A_T_line[i];
                        // float Ai = A[(it * TILE_SIZE4 + i) * CFG::ffdim + kt * TILE_SIZE4 + k];
                        for (int j = 0; j < TILE_SIZE4_J; ++j)
                        {
#pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }

            add_skip_stream(it, jt, out_block, skip_buff, out);
        }
    }
}

void linear_dataflow2(float *in_buff, float *weight, float *bias, float *out_buff, float *skip)
{
#pragma HLS dataflow
    static hls::stream<float> dense_weight_stream("dense_weight_stream");
    static hls::stream<float> dense_bias_stream("dense_bias_stream");
    static hls::stream<float> skip_stream("skip_stream");

#pragma HLS stream variable = dense_weight_stream depth = 128
#pragma HLS stream variable = dense_bias_stream depth = 128
#pragma HLS stream variable = skip_stream depth = 128

    read_weights1(weight, dense_weight_stream);
    read_bias1(bias, dense_bias_stream);
    read_skip2(skip, skip_stream);

    linear_fused2(in_buff, dense_weight_stream, dense_bias_stream, out_buff, skip_stream);
}

void linear_dataflow4(float *in, float *dense_weight_t, float *dense_bias, float *out_buff, float *skip_conn)
{

    static hls::stream<float> A_stream("A_stream");
    static hls::stream<float> B_stream("B_stream");
    static hls::stream<float> bias_stream("bias_stream");
    static hls::stream<float> skip_stream("skip_stream");

#pragma HLS stream variable = A_stream depth = 128
#pragma HLS stream variable = B_stream depth = 128
#pragma HLS stream variable = bias_stream depth = 128
#pragma HLS stream variable = skip_stream depth = 128
    // #pragma HLS dataflow

    read_A4(in, A_stream);
    read_B4(dense_weight_t, B_stream);
    read_bias4(dense_bias, bias_stream);
    read_skip4(skip_conn, skip_stream);
    linear_fused4(A_stream, B_stream, bias_stream, out_buff, skip_stream);
}

void attention_values_fused(float *probs, float *value, float *attn_out)
{

    /**
     * probs: <nhead, seqlen, seqlen>
     * value: <seqlen, dmodel> -> <nhead, seqlen, dhead> (same reshape/transpose as query)
     *
     * attn_out: <nhead, seqlen, dhead> -> <seqlen, dmodel> (how do you index to do this in one shot)
     * attn_out[i1][i2][i3] = attn_out[i1*seqlen*dhead + i2*dhead + i3]
     * att_out_transpose is <seqlen, nhead, dhead>
     * att_out_transpose[i1][i2][i3] = attn_out[i2][i1][i3] = attn_out[i2*nhead*dhead + i1*dhead + i3]
     *
     * value_transpose[i1][i2][i3] = value[i2*nhead*dhead +i1*dhead + i3]
     *
     */

    float row_buf[CFG::dhead];
    float probs_row[CFG::seqlen];
    float value_row[CFG::dhead];
#pragma HLS array_partition variable = row_buf complete
#pragma HLS array_partition variable = value_row complete

    for (int n = 0; n < CFG::nhead; n++)
    {
        for (int i = 0; i < CFG::seqlen; i++)
        {
            for (int k = 0; k < CFG::seqlen; ++k)
            {
                probs_row[k] = probs[n * CFG::seqlen * CFG::seqlen + i * CFG::seqlen + k];
            }
            for (int j = 0; j < CFG::dhead; ++j)
            {
                row_buf[j] = 0;
            }
            for (int k = 0; k < CFG::seqlen; k++)
            {
                for (int j = 0; j < CFG::dhead; ++j)
                {
                    value_row[j] = value[k * CFG::nhead * CFG::dhead + n * CFG::dhead + j];
                }
                float probs_k = probs_row[k];
                for (int j = 0; j < CFG::dhead; j++)
                {
#pragma HLS unroll
                    // attn_out[n][i][j] += probs[n][i][k] * value[n][k][j]
                    row_buf[j] += probs_k * value_row[j];
                }
            }
            for (int j = 0; j < CFG::dhead; ++j)
            {
                attn_out[i * CFG::nhead * CFG::dhead + n * CFG::dhead + j] = float(row_buf[j]);
            }
        }
    }
}

void attention_layer(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                     float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *out)
{

    auto query_in = new float[CFG::seqlen * CFG::dmodel];
    auto key_in = new float[CFG::seqlen * CFG::dmodel];
    auto value_in = new float[CFG::seqlen * CFG::dmodel];
    auto attn_score = new float[CFG::nhead * CFG::seqlen * CFG::seqlen];
    auto attn_probs = new float[CFG::nhead * CFG::seqlen * CFG::seqlen];
    auto attn_out = new float[CFG::seqlen * CFG::dmodel];
    auto dense_out = new float[CFG::seqlen * CFG::dmodel];

    linear_sw_base(in, query_weight_t, query_bias, query_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw_base(in, key_weight_t, key_bias, key_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw_base(in, value_weight_t, value_bias, value_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);

    attention_scores(query_in, key_in, attn_score, CFG::seqlen, CFG::nhead, CFG::dhead);
    softmax_nn(attn_score, attn_probs);
    attention_values(attn_probs, value_in, attn_out, CFG::seqlen, CFG::nhead, CFG::dhead);

    linear_sw_base(attn_out, dense_weight_t, dense_bias, dense_out, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    add_skip2(dense_out, in, CFG::seqlen * CFG::dmodel);
    layernorm_base(dense_out, out, norm_weight, norm_bias);
}

void FFN(float *fc_in, float *dense_weight_t, float *dense_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out)
{

    // Refactor to fuse layers and not dynamically allocate these
    auto dense_temp = new float[CFG::seqlen * CFG::ffdim];
    auto dense_temp1 = new float[CFG::seqlen * CFG::dmodel];
    auto gelu_temp = new float[CFG::seqlen * CFG::ffdim];
    float dense_acc_scale = 0.004;
    linear_sw_base(fc_in, dense_weight_t, dense_bias, dense_temp, CFG::seqlen, CFG::ffdim, CFG::dmodel);
    // printf("ffn1 test = : \n");
    // for (int i = 0; i < CFG::seqlen; ++i)
    // {
    //     for (int j = 0; j < CFG::ffdim; ++j)
    //     {

    //         printf("%f ", dense_temp[i * CFG::seqlen + j]);
    //     }
    //     printf("\n");
    // }
    gelu_base(dense_temp, gelu_temp, CFG::seqlen, CFG::ffdim);
    linear_sw_base(gelu_temp, dense2_weight_t, dense2_bias, dense_temp1, CFG::seqlen, CFG::dmodel, CFG::ffdim);
    add_skip2(dense_temp1, fc_in, CFG::seqlen * CFG::dmodel);
    layernorm_base(dense_temp1, dense_out, ffn_norm_weight, ffn_norm_bias);
}

/*hardware*/

void attention_kernel(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                      float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *out)
{

// Can run all linear layers in parallel
#pragma HLS dataflow

    static hls::stream<float> in_query("in_query_stream");
    static hls::stream<float> in_key("in_key_stream");
    static hls::stream<float> in_value("in_value_stream");

    static hls::stream<float> query_weight_stream("query_weight_stream");
    static hls::stream<float> key_weight_stream("key_weight_stream");
    static hls::stream<float> value_weight_stream("value_weight_stream");

    static hls::stream<float> query_bias_stream("query_bias_stream");
    static hls::stream<float> key_bias_stream("key_bias_stream");
    static hls::stream<float> value_bias_stream("value_bias_stream");

    static hls::stream<float> query_out_stream("query_out_stream");
    static hls::stream<float> key_out_stream("key_out_stream");
    static hls::stream<float> value_out_stream("value_out_stream");
#pragma HLS stream variable = in_query depth = 128
#pragma HLS stream variable = in_key depth = 128
#pragma HLS stream variable = in_value depth = 128

#pragma HLS stream variable = query_weight_stream depth = 128
#pragma HLS stream variable = key_weight_stream depth = 128
#pragma HLS stream variable = value_weight_stream depth = 128

#pragma HLS stream variable = query_bias_stream depth = 128
#pragma HLS stream variable = key_bias_stream depth = 128
#pragma HLS stream variable = value_bias_stream depth = 128

    // #pragma HLS stream variable = query_out_stream depth = 128
    // #pragma HLS stream variable = key_out_stream depth = 128

    float query_out[CFG::seqlen * CFG::dmodel];
    float key_out[CFG::seqlen * CFG::dmodel];
    float value_out[CFG::seqlen * CFG::dmodel];
    float attn_score[CFG::nhead * CFG::seqlen * CFG::seqlen];
    float attn_probs[CFG::nhead * CFG::seqlen * CFG::seqlen];
    float attn_out[CFG::seqlen * CFG::dmodel];
    float dense_out[CFG::seqlen * CFG::dmodel];
    float attn_score_gt[CFG::nhead * CFG::seqlen * CFG::seqlen];
    float in_a[CFG::seqlen * CFG::dmodel];

    read_input(in, in_query, in_key, in_value, in_a);

    read_weights1(query_weight_t, query_weight_stream);
    read_weights1(key_weight_t, key_weight_stream);
    read_weights1(value_weight_t, value_weight_stream);

    read_bias1(query_bias, query_bias_stream);
    read_bias1(key_bias, key_bias_stream);
    read_bias1(value_bias, value_bias_stream);

    linear_fused1(in_query, query_weight_stream, query_bias_stream, query_out_stream);
    linear_fused1(in_key, key_weight_stream, key_bias_stream, key_out_stream);
    linear_fused1(in_value, value_weight_stream, value_bias_stream, value_out_stream);

    write_out1(query_out, query_out_stream);
    write_out1(key_out, key_out_stream);
    write_out1(value_out, value_out_stream);

    attention_scores_fused(query_out, key_out, attn_score);
    // attention_scores_fused_stream(query_out_stream, key_out_stream, attn_score);
    attention_values_fused(attn_score, value_out, attn_out);

    // linear,  residual
    linear_dataflow2(attn_out, dense_weight_t, dense_bias, dense_out, in_a);

    layernorm_base(dense_out, out, norm_weight, norm_bias);
}

float gelu_fused(float gelu_in)
{
    // float PI = 3.14159265358979323846;
    float PI = 3.1416;
    float A = 0.044715;
    float a = sqrt(2.0 / PI);

    float val = gelu_in;
    float pow3 = val * val * val;
    float b = val + A * pow3;

    val = 0.5 * val * (1 + tanh(a * b));
    gelu_in = val;

    return float(gelu_in);
}

void linear_fused(hls::stream<float> &A_stream, hls::stream<float> &B_stream, hls::stream<float> &bias_stream, hls::stream<float> &out_stream)
{
    // compute fused gelu constants

    // int_erf

    // buffers for tile mmult
    float out_block[TILE_SIZE][TILE_SIZE_J];
    float B_line[TILE_SIZE_J];
    float A_T_line[TILE_SIZE];

#pragma HLS array_partition dim = 2 complete variable = out_block
//#pragma HLS array_partition dim=1 type=cyclic factor=32 variable=out_block
#pragma HLS array_partition dim = 1 complete variable = B_line

    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
#pragma HLS unroll
        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            // #pragma HLS unroll
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    out_block[i][j] = bias_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE; ++kt)
            {
                for (int k = 0; k < TILE_SIZE; ++k)
                {

                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE_J; ++j)
                    {
                        B_line[j] = B_stream.read();
                    }
                    for (int i = 0; i < TILE_SIZE; ++i)
                    {
                        A_T_line[i] = A_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE; ++i)
                    {
//#pragma HLS unroll factor=4
#pragma HLS PIPELINE
                        float Ai = A_T_line[i];
                        for (int j = 0; j < TILE_SIZE_J; ++j)
                        {
#pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }

            // apply gelu and write output
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    float a = gelu_fused(out_block[i][j]);
                    out_stream.write(a);
                }
            }
        }
    }
}

void linear_fused_temp(hls::stream<float> &A_stream, hls::stream<float> &B_stream, hls::stream<float> &bias_stream, hls::stream<float> &out_stream)
{
    // compute fused gelu constants

    // int_erf

    // buffers for tile mmult
    float out_block[CFG::ffdim / TILE_SIZE_J][TILE_SIZE][TILE_SIZE_J];
    float B_line[CFG::dmodel / TILE_SIZE][TILE_SIZE_J];
    float A_T_line[CFG::dmodel / TILE_SIZE][TILE_SIZE];
    float Ai[CFG::dmodel / TILE_SIZE][TILE_SIZE];
#pragma HLS array_partition dim = 3 complete variable = out_block
#pragma HLS array_partition dim = 1 complete variable = out_block
#pragma HLS array_partition dim = 2 complete variable = B_line
#pragma HLS array_partition dim = 1 complete variable = B_line
#pragma HLS array_partition dim = 2 complete variable = A_T_line
#pragma HLS array_partition dim = 1 complete variable = A_T_line
#pragma HLS array_partition dim = 2 complete variable = Ai
#pragma HLS array_partition dim = 1 complete variable = Ai

    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {

        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            // #pragma HLS unroll
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    out_block[jt][i][j] = bias_stream.read();
                }
            }

            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE; ++kt)
            {
#pragma HLS unroll
                for (int k = 0; k < TILE_SIZE; ++k)
                {

                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE_J; ++j)
                    {
                        B_line[kt][j] = B_stream.read();
                    }
                    for (int i = 0; i < TILE_SIZE; ++i)
                    {
                        A_T_line[kt][i] = A_stream.read();
                    }

                    for (int i = 0; i < TILE_SIZE; ++i)
                    {
//#pragma HLS unroll factor=4
#pragma HLS PIPELINE
                        Ai[kt][i] = A_T_line[kt][i];
                        for (int j = 0; j < TILE_SIZE_J; ++j)
                        {
#pragma HLS unroll
                            out_block[jt][i][j] += Ai[kt][i] * B_line[kt][j];
                        }
                    }
                }
            }

            // apply gelu and write output
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    float a = gelu_fused(out_block[jt][i][j]);
                    out_stream.write(a);
                }
            }
        }
    }
}

void FFN_dense1(float *fc_in, float *dense_weight_t, float *dense_bias, float *dense_out)
{
    static hls::stream<float> A_stream("A_stream");
    static hls::stream<float> B_stream("B_stream");
    static hls::stream<float> bias_stream("bias_stream");
    static hls::stream<float> out_stream("out_stream");

#pragma HLS stream variable = A_stream depth = 128
#pragma HLS stream variable = B_stream depth = 128
#pragma HLS stream variable = bias_stream depth = 128
#pragma HLS stream variable = out_stream depth = 128

#pragma HLS dataflow

    read_A(fc_in, A_stream);
    read_B(dense_weight_t, B_stream);
    read_bias(dense_bias, bias_stream);
    linear_fused(A_stream, B_stream, bias_stream, out_stream);
    write_out(dense_out, out_stream);
}

void FFN_kernel(float *fc_in, float *dense_weight_t, float *dense_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out)
{

    // Refactor to fuse layers and not dynamically allocate these
    float dense_temp[CFG::seqlen * CFG::ffdim];

    FFN_dense1(fc_in, dense_weight_t, dense_bias, dense_temp);
    // linear,  residual
    linear_dataflow4(dense_temp, dense2_weight_t, dense2_bias, dense_temp, fc_in);
    // layernorm
    layernorm_base(dense_temp, dense_out, ffn_norm_weight, ffn_norm_bias);
}

void qkv(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
         float *query_in, float *key_in, float *value_in)
{

    linear_sw_base(in, query_weight_t, query_bias, query_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw_base(in, key_weight_t, key_bias, key_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw_base(in, value_weight_t, value_bias, value_in, CFG::seqlen, CFG::dmodel, CFG::dmodel);
}

void encoder(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
             float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *dense1_weight_t,
             float *dense1_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out)
{

    auto fc_in = new float[CFG::seqlen * CFG::dmodel];

    attention_layer(in, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
                    dense_weight_t, dense_bias, norm_weight, norm_bias, fc_in);

    FFN(fc_in, dense1_weight_t, dense1_bias, dense2_weight_t, dense2_bias, ffn_norm_weight, ffn_norm_bias, dense_out);
}

void encoder_kernel(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                    float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *dense1_weight_t,
                    float *dense1_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out)
{

    // #pragma HLS interface m_axi port = in bundle = gmem0
    // #pragma HLS interface m_axi port = query_weight_t bundle = gmem1
    // #pragma HLS interface m_axi port = query_bias bundle = gmem2
    // #pragma HLS interface m_axi port = key_weight_t bundle = gmem3
    // #pragma HLS interface m_axi port = key_bias bundle = gmem4
    // #pragma HLS interface m_axi port = value_weight_t bundle = gmem5
    // #pragma HLS interface m_axi port = value_bias bundle = gmem6

    // #pragma HLS interface m_axi port = dense_weight_t bundle = gmem7
    // #pragma HLS interface m_axi port = dense_bias bundle = gmem8
    // #pragma HLS interface m_axi port = norm_weight bundle = gmem9
    // #pragma HLS interface m_axi port = norm_bias bundle = gmem10

    // #pragma HLS interface m_axi port = dense1_weight_t bundle = gmem11
    // #pragma HLS interface m_axi port = dense1_bias bundle = gmem12
    // #pragma HLS interface m_axi port = dense2_weight_t bundle = gmem13
    // #pragma HLS interface m_axi port = dense2_bias bundle = gmem14
    // #pragma HLS interface m_axi port = ffn_norm_weight bundle = gmem15
    // #pragma HLS interface m_axi port = ffn_norm_bias bundle = gmem16
    // #pragma HLS interface m_axi port = dense_out bundle = gmem17

    // #pragma HLS interface s_axilite port = return bundle = control

    float fc_in[CFG::seqlen * CFG::dmodel];
    float fc_in_T[CFG::seqlen * CFG::dmodel];

    attention_kernel(in, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
                     dense_weight_t, dense_bias, norm_weight, norm_bias, fc_in);

    matrix2d_transpose(fc_in, fc_in_T, CFG::dmodel, CFG::seqlen);
    FFN_kernel(fc_in_T, dense1_weight_t, dense1_bias, dense2_weight_t, dense2_bias, ffn_norm_weight, ffn_norm_bias, dense_out);
}
