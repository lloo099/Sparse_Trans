#include "config.hpp"
#include "hls_stream.h"
const int TILE_SIZE1 = 256;
const int TILE_SIZE2 = 256;
const int TILE_SIZE3 = 256;
const int TILE_SIZE4 = 256;
const int TILE_SIZE_J = 256;
const int TILE_SIZE4_J = 256;
const int TILE_SIZE = 256;

// void read_input1(float *in, hls::stream<float> &in_1, hls::stream<float> &in_2, hls::stream<float> &in_3, float *in_tem)
// {

//     float in_T[CFG::seqlen * CFG::dmodel];

//     for (int i = 0; i < CFG::dmodel; ++i)
//     {
//         for (int j = 0; j < CFG::seqlen; ++j)
//         {
//             in_T[i * CFG::seqlen + j] = in[j * CFG::dmodel + i];
//         }
//     }

//     for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it)
//     {
//         for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt)
//         {
//             for (int kt = 0; kt < CFG::dmodel / TILE_SIZE1; ++kt)
//             {
//                 for (int k = 0; k < TILE_SIZE1; ++k)
//                 {
//                     for (int i = 0; i < TILE_SIZE1; ++i)
//                     {
//                         float A_val = in_T[(kt * TILE_SIZE1 + k) * CFG::seqlen + it * TILE_SIZE1 + i];
//                         in_1.write(A_val);
//                         in_2.write(A_val);
//                         in_3.write(A_val);
//                         in_tem[(kt * TILE_SIZE1 + k) * CFG::seqlen + it * TILE_SIZE1 + i] = in_T[(kt * TILE_SIZE1 + k) * CFG::seqlen + it * TILE_SIZE1 + i];
//                     }
//                 }
//             }
//         }
//     }
// }

void read_weights1(float *weights, hls::stream<float> &weights_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt)
        {
            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE1; ++kt)
            {
                for (int k = 0; k < TILE_SIZE1; ++k)
                {
                    for (int j = 0; j < TILE_SIZE1; ++j)
                    {
#pragma HLS PIPELINE II = 1
                        weights_stream.write(weights[(kt * TILE_SIZE1 + k) * CFG::dmodel + jt * TILE_SIZE1 + j]);
                    }
                }
            }
        }
    }
}

void read_bias1(float *bias, hls::stream<float> &bias_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1; ++j)
                {
#pragma HLS PIPELINE II = 1
                    bias_stream.write(bias[jt * TILE_SIZE1 + j]);
                }
            }
        }
    }
}

void read_skip2(float *skip, hls::stream<float> &skip_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE2; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE2; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE2; ++i)
            {
                for (int j = 0; j < TILE_SIZE2; ++j)
                {
#pragma HLS PIPELINE II = 1
                    skip_stream.write(skip[(jt * TILE_SIZE2 + j) * CFG::seqlen + it * TILE_SIZE2 + i]);
                }
            }
        }
    }
}

void write_out1(float *out, hls::stream<float> &out_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt)
        {
            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1; ++j)
                {
#pragma HLS PIPELINE II = 1
                    out[(it * TILE_SIZE1 + i) * CFG::dmodel + jt * TILE_SIZE1 + j] = out_stream.read();
                }
            }
        }
    }
}

void read_A(float *A, hls::stream<float> &A_stream)
{

    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE; ++kt)
            {
                for (int k = 0; k < TILE_SIZE; ++k)
                {
                    for (int i = 0; i < TILE_SIZE; ++i)
                    {
#pragma HLS PIPELINE II = 1
                        A_stream.write(A[(kt * TILE_SIZE + k) * CFG::seqlen + it * TILE_SIZE + i]);
                    }
                }
            }
        }
    }
}

void read_B(float *B, hls::stream<float> &B_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            for (int kt = 0; kt < CFG::dmodel / TILE_SIZE; ++kt)
            {
                for (int k = 0; k < TILE_SIZE; ++k)
                {
                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE_J; ++j)
                    {
#pragma HLS PIPELINE II = 1
                        B_stream.write(B[(kt * TILE_SIZE + k) * CFG::ffdim + jt * TILE_SIZE_J + j]);
                    }
                }
            }
        }
    }
}

void read_bias(float *bias, hls::stream<float> &bias_stream)
{

    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
#pragma HLS PIPELINE II = 1
                    bias_stream.write(bias[jt * TILE_SIZE_J + j]);
                }
            }
        }
    }
}

void write_out(float *out_T, hls::stream<float> &out_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::ffdim / TILE_SIZE_J; ++jt)
        {
            // apply gelu and write output
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    out_T[(jt * TILE_SIZE_J + j) * CFG::seqlen + it * TILE_SIZE + i] = out_stream.read();
                }
            }
        }
    }
}

void read_A4(float *A, hls::stream<float> &A_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE4; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE4_J; ++jt)
        {
            for (int kt = 0; kt < CFG::ffdim / TILE_SIZE4; ++kt)
            {
                for (int k = 0; k < TILE_SIZE4; ++k)
                {
                    for (int i = 0; i < TILE_SIZE4; ++i)
                    {
                        A_stream.write(A[(kt * TILE_SIZE4 + k) * CFG::seqlen + it * TILE_SIZE4 + i]);
                    }
                }
            }
        }
    }
}

void read_B4(float *B, hls::stream<float> &B_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE4; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE4_J; ++jt)
        {
            for (int kt = 0; kt < CFG::ffdim / TILE_SIZE4; ++kt)
            {
                for (int k = 0; k < TILE_SIZE4; ++k)
                {
                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE4_J; ++j)
                    {
                        B_stream.write(B[(kt * TILE_SIZE4 + k) * CFG::dmodel + jt * TILE_SIZE4_J + j]);
                    }
                }
            }
        }
    }
}

void read_bias4(float *bias, hls::stream<float> &bias_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE4; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE4_J; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE4; ++i)
            {
                for (int j = 0; j < TILE_SIZE4_J; ++j)
                {
                    bias_stream.write(bias[jt * TILE_SIZE4_J + j]);
                }
            }
        }
    }
}

void read_skip4(float *skip_conn, hls::stream<float> &skip_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE4; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE4_J; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE4; ++i)
            {
                for (int j = 0; j < TILE_SIZE4_J; ++j)
                {
                    skip_stream.write(skip_conn[(jt * TILE_SIZE4_J + j) * CFG::seqlen + it * TILE_SIZE4 + i]);
                }
            }
        }
    }
}

void write_out4(float *out_T, hls::stream<float> &out_stream)
{
    for (int it = 0; it < CFG::seqlen / TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE_J; ++jt)
        {
            // apply gelu and write output
            for (int i = 0; i < TILE_SIZE; ++i)
            {
                for (int j = 0; j < TILE_SIZE_J; ++j)
                {
                    out_T[(jt * TILE_SIZE_J + j) * CFG::seqlen + it * TILE_SIZE + i] = out_stream.read();
                }
            }
        }
    }
}