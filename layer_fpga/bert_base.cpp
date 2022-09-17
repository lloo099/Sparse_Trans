#include "config.hpp"
#include "layer.hpp"
#include "parameters.hpp"
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

void printmat(float *A, const int M, const int N)
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

void bert_base(float *input_ids, float *token_ids, float *pos_ids, float *layer_out)
{

        // #pragma HLS INTERFACE axis port = input_ids, token_ids, pos_ids, layer_out

        auto dense_encoder1 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder2 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder3 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder4 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder5 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder6 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder7 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder8 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder9 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder10 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder11 = new float[CFG::seqlen * CFG::dmodel];
        auto dense_encoder12 = new float[CFG::seqlen * CFG::dmodel];

        /**
         * embedding load
         */
        load_from_txt<float, CFG::word * CFG::dmodel>(word_embeddings, "word_embedding_hls.txt");
        load_from_txt<float, CFG::pos * CFG::dmodel>(pos_embeddings, "pos_embedding_hls.txt");
        load_from_txt<float, CFG::token * CFG::dmodel>(token_embeddings, "token_embedding_hls.txt");
        load_from_txt<float, CFG::dmodel>(embed_norm_weight, "embed_norm_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(embed_norm_bias, "embed_norm_bias_hls.txt");

        /**
         * encoder 1 load
         */
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

        /**
         * encoder 2 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query1_weight_t, "encoder2_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key1_weight_t, "encoder2_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value1_weight_t, "encoder2_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense1_weight_t, "encoder2_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm1_weight, "encoder2_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query1_bias, "encoder2_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key1_bias, "encoder2_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value1_bias, "encoder2_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense1_bias, "encoder2_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm1_bias, "encoder2_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn1_dense_weight_t, "encoder2_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn1_dense_bias, "encoder2_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn1_dense2_weight_t, "encoder2_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_dense2_bias, "encoder2_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_norm_weight, "encoder2_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_norm_bias, "encoder2_ffn2_lnbias_hls.txt");

        /**
         * encoder 3 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query2_weight_t, "encoder3_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key2_weight_t, "encoder3_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value2_weight_t, "encoder3_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense2_weight_t, "encoder3_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm2_weight, "encoder3_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query2_bias, "encoder3_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key2_bias, "encoder3_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value2_bias, "encoder3_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense2_bias, "encoder3_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm2_bias, "encoder3_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn2_dense_weight_t, "encoder3_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn2_dense_bias, "encoder3_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn2_dense2_weight_t, "encoder3_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_dense2_bias, "encoder3_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_norm_weight, "encoder3_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_norm_bias, "encoder3_ffn2_lnbias_hls.txt");

        /**
         * encoder 4 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query3_weight_t, "encoder4_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key3_weight_t, "encoder4_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value3_weight_t, "encoder4_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense3_weight_t, "encoder4_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm3_weight, "encoder4_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query3_bias, "encoder4_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key3_bias, "encoder4_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value3_bias, "encoder4_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense3_bias, "encoder4_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm3_bias, "encoder4_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn3_dense_weight_t, "encoder4_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn3_dense_bias, "encoder4_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn3_dense2_weight_t, "encoder4_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_dense2_bias, "encoder4_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_norm_weight, "encoder4_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_norm_bias, "encoder4_ffn2_lnbias_hls.txt");

        /**
         * encoder 5 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query4_weight_t, "encoder5_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key4_weight_t, "encoder5_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value4_weight_t, "encoder5_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense4_weight_t, "encoder5_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm4_weight, "encoder5_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query4_bias, "encoder5_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key4_bias, "encoder5_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value4_bias, "encoder5_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense4_bias, "encoder5_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm4_bias, "encoder5_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn4_dense_weight_t, "encoder5_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn4_dense_bias, "encoder5_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn4_dense2_weight_t, "encoder5_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_dense2_bias, "encoder5_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_norm_weight, "encoder5_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_norm_bias, "encoder5_ffn2_lnbias_hls.txt");

        /**
         * encoder 6 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query5_weight_t, "encoder6_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key5_weight_t, "encoder6_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value5_weight_t, "encoder6_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense5_weight_t, "encoder6_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm5_weight, "encoder6_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query5_bias, "encoder6_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key5_bias, "encoder6_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value5_bias, "encoder6_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense5_bias, "encoder6_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm5_bias, "encoder6_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn5_dense_weight_t, "encoder6_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn5_dense_bias, "encoder6_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn5_dense2_weight_t, "encoder6_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_dense2_bias, "encoder6_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_norm_weight, "encoder6_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_norm_bias, "encoder6_ffn2_lnbias_hls.txt");

        /**
         * encoder 7 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query6_weight_t, "encoder7_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key6_weight_t, "encoder7_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value6_weight_t, "encoder7_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense6_weight_t, "encoder7_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm6_weight, "encoder7_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query6_bias, "encoder7_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key6_bias, "encoder7_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value6_bias, "encoder7_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense6_bias, "encoder7_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm6_bias, "encoder7_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn6_dense_weight_t, "encoder7_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn6_dense_bias, "encoder7_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn6_dense2_weight_t, "encoder7_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_dense2_bias, "encoder7_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_norm_weight, "encoder7_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_norm_bias, "encoder7_ffn2_lnbias_hls.txt");

        /**
         * encoder 8 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query7_weight_t, "encoder8_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key7_weight_t, "encoder8_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value7_weight_t, "encoder8_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense7_weight_t, "encoder8_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm7_weight, "encoder8_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query7_bias, "encoder8_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key7_bias, "encoder8_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value7_bias, "encoder8_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense7_bias, "encoder8_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm7_bias, "encoder8_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn7_dense_weight_t, "encoder8_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn7_dense_bias, "encoder8_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn7_dense2_weight_t, "encoder8_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_dense2_bias, "encoder8_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_norm_weight, "encoder8_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_norm_bias, "encoder8_ffn2_lnbias_hls.txt");

        /**
         * encoder 9 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query8_weight_t, "encoder9_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key8_weight_t, "encoder9_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value8_weight_t, "encoder9_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense8_weight_t, "encoder9_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm8_weight, "encoder9_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query8_bias, "encoder9_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key8_bias, "encoder9_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value8_bias, "encoder9_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense8_bias, "encoder9_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm8_bias, "encoder9_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn8_dense_weight_t, "encoder9_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn8_dense_bias, "encoder9_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn8_dense2_weight_t, "encoder9_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_dense2_bias, "encoder9_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_norm_weight, "encoder9_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_norm_bias, "encoder9_ffn2_lnbias_hls.txt");

        /**
         * encoder 10 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query9_weight_t, "encoder10_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key9_weight_t, "encoder10_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value9_weight_t, "encoder10_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense9_weight_t, "encoder10_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm9_weight, "encoder10_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query9_bias, "encoder10_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key9_bias, "encoder10_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value9_bias, "encoder10_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense9_bias, "encoder10_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm9_bias, "encoder10_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn9_dense_weight_t, "encoder10_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn9_dense_bias, "encoder10_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn9_dense2_weight_t, "encoder10_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_dense2_bias, "encoder10_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_norm_weight, "encoder10_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_norm_bias, "encoder10_ffn2_lnbias_hls.txt");

        /**
         * encoder 11 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query10_weight_t, "encoder11_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key10_weight_t, "encoder11_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value10_weight_t, "encoder11_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense10_weight_t, "encoder11_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm10_weight, "encoder11_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query10_bias, "encoder11_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key10_bias, "encoder11_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value10_bias, "encoder11_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense10_bias, "encoder11_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm10_bias, "encoder11_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn10_dense_weight_t, "encoder11_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn10_dense_bias, "encoder11_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn10_dense2_weight_t, "encoder11_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_dense2_bias, "encoder11_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_norm_weight, "encoder11_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_norm_bias, "encoder11_ffn2_lnbias_hls.txt");

        /**
         * encoder 12 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query11_weight_t, "encoder12_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key11_weight_t, "encoder12_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value11_weight_t, "encoder12_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense11_weight_t, "encoder12_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm11_weight, "encoder12_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query11_bias, "encoder12_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key11_bias, "encoder12_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value11_bias, "encoder12_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense11_bias, "encoder12_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm11_bias, "encoder12_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn11_dense_weight_t, "encoder12_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn11_dense_bias, "encoder12_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn11_dense2_weight_t, "encoder12_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_dense2_bias, "encoder12_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_norm_weight, "encoder12_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_norm_bias, "encoder12_ffn2_lnbias_hls.txt");

        load_from_txt<float, CFG::dmodel * CFG::dmodel>(pooler_weight, "pooler_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(pooler_bias, "pooler_bias_hls.txt");

        embedding_out(word_embeddings, token_embeddings, pos_embeddings, input_ids, token_ids, pos_ids, embed_norm_weight, embed_norm_bias, embed_out);
        encoder(embed_out, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
                dense_weight_t, dense_bias, norm_weight, norm_bias, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t,
                ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, dense_encoder1);
        printf("encoder1 test = : \n");
        printmat(dense_encoder1, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder1, query1_weight_t, query1_bias, key1_weight_t, key1_bias, value1_weight_t, value1_bias,
                dense1_weight_t, dense1_bias, norm1_weight, norm1_bias, ffn1_dense_weight_t, ffn1_dense_bias, ffn1_dense2_weight_t,
                ffn1_dense2_bias, ffn1_norm_weight, ffn1_norm_bias, dense_encoder2);
        printf("encoder2 test = : \n");
        printmat(dense_encoder2, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder2, query2_weight_t, query2_bias, key2_weight_t, key2_bias, value2_weight_t, value2_bias,
                dense2_weight_t, dense2_bias, norm2_weight, norm2_bias, ffn2_dense_weight_t, ffn2_dense_bias, ffn2_dense2_weight_t,
                ffn2_dense2_bias, ffn2_norm_weight, ffn2_norm_bias, dense_encoder3);
        printf("encoder3 test = : \n");
        printmat(dense_encoder3, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder3, query3_weight_t, query3_bias, key3_weight_t, key3_bias, value3_weight_t, value3_bias,
                dense3_weight_t, dense3_bias, norm3_weight, norm3_bias, ffn3_dense_weight_t, ffn3_dense_bias, ffn3_dense2_weight_t,
                ffn3_dense2_bias, ffn3_norm_weight, ffn3_norm_bias, dense_encoder4);
        printf("encoder4 test = : \n");
        printmat(dense_encoder4, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder4, query4_weight_t, query4_bias, key4_weight_t, key4_bias, value4_weight_t, value4_bias,
                dense4_weight_t, dense4_bias, norm4_weight, norm4_bias, ffn4_dense_weight_t, ffn4_dense_bias, ffn4_dense2_weight_t,
                ffn4_dense2_bias, ffn4_norm_weight, ffn4_norm_bias, dense_encoder5);
        printf("encoder5 test = : \n");
        printmat(dense_encoder5, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder5, query5_weight_t, query5_bias, key5_weight_t, key5_bias, value5_weight_t, value5_bias,
                dense5_weight_t, dense5_bias, norm5_weight, norm5_bias, ffn5_dense_weight_t, ffn5_dense_bias, ffn5_dense2_weight_t,
                ffn5_dense2_bias, ffn5_norm_weight, ffn5_norm_bias, dense_encoder6);
        printf("encoder6 test = : \n");
        printmat(dense_encoder6, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder6, query6_weight_t, query6_bias, key6_weight_t, key6_bias, value6_weight_t, value6_bias,
                dense6_weight_t, dense6_bias, norm6_weight, norm6_bias, ffn6_dense_weight_t, ffn6_dense_bias, ffn6_dense2_weight_t,
                ffn6_dense2_bias, ffn6_norm_weight, ffn6_norm_bias, dense_encoder7);
        printf("encoder7 test = : \n");
        printmat(dense_encoder7, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder7, query7_weight_t, query7_bias, key7_weight_t, key7_bias, value7_weight_t, value7_bias,
                dense7_weight_t, dense7_bias, norm7_weight, norm7_bias, ffn7_dense_weight_t, ffn7_dense_bias, ffn7_dense2_weight_t,
                ffn7_dense2_bias, ffn7_norm_weight, ffn7_norm_bias, dense_encoder8);
        printf("encoder8 test = : \n");
        printmat(dense_encoder8, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder8, query8_weight_t, query8_bias, key8_weight_t, key8_bias, value8_weight_t, value8_bias,
                dense8_weight_t, dense8_bias, norm8_weight, norm8_bias, ffn8_dense_weight_t, ffn8_dense_bias, ffn8_dense2_weight_t,
                ffn8_dense2_bias, ffn8_norm_weight, ffn8_norm_bias, dense_encoder9);
        printf("encoder9 test = : \n");
        printmat(dense_encoder9, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder9, query9_weight_t, query9_bias, key9_weight_t, key9_bias, value9_weight_t, value9_bias,
                dense9_weight_t, dense9_bias, norm9_weight, norm9_bias, ffn9_dense_weight_t, ffn9_dense_bias, ffn9_dense2_weight_t,
                ffn9_dense2_bias, ffn9_norm_weight, ffn9_norm_bias, dense_encoder10);
        printf("encoder10 test = : \n");
        printmat(dense_encoder10, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder10, query10_weight_t, query10_bias, key10_weight_t, key10_bias, value10_weight_t, value10_bias,
                dense10_weight_t, dense10_bias, norm10_weight, norm10_bias, ffn10_dense_weight_t, ffn10_dense_bias, ffn10_dense2_weight_t,
                ffn10_dense2_bias, ffn10_norm_weight, ffn10_norm_bias, dense_encoder11);
        printf("encoder11 test = : \n");
        printmat(dense_encoder11, CFG::seqlen, CFG::dmodel);

        encoder(dense_encoder11, query11_weight_t, query11_bias, key11_weight_t, key11_bias, value11_weight_t, value11_bias,
                dense11_weight_t, dense11_bias, norm11_weight, norm11_bias, ffn11_dense_weight_t, ffn11_dense_bias, ffn11_dense2_weight_t,
                ffn11_dense2_bias, ffn11_norm_weight, ffn11_norm_bias, dense_encoder12);
        printf("encoder12 test = : \n");
        printmat(dense_encoder12, CFG::seqlen, CFG::dmodel);

        for (int i = 0; i < CFG::dmodel; ++i)
        {
                for (int j = 0; j < CFG::dmodel; ++j)
                {
                        pooler_weight_T[i * CFG::dmodel + j] = pooler_weight[j * CFG::dmodel + i];
                        // printf("%f ", hidden_states_T[i * CFG::seqlen + j]);
                }
                // printf("\n");
        }

        linear_pooler(dense_encoder12, pooler_weight_T, pooler_bias, layer_out, 1, CFG::dmodel, CFG::dmodel);
}

void bert_fpga1(float *input_ids, float *token_ids, float *pos_ids, float *layer_out)
{

// #pragma HLS INTERFACE axis port = input_ids, token_ids, pos_ids, layer6_out
#pragma HLS interface m_axi port = input_ids bundle = gmem0
#pragma HLS interface m_axi port = token_ids bundle = gmem1
#pragma HLS interface m_axi port = pos_ids bundle = gmem2
#pragma HLS interface m_axi port = layer_out bundle = gmem3

#pragma HLS interface s_axilite port = return bundle = control

        float dense_encoder1[CFG::seqlen * CFG::dmodel];
        float dense_encoder2[CFG::seqlen * CFG::dmodel];
        float dense_encoder3[CFG::seqlen * CFG::dmodel];
        float dense_encoder4[CFG::seqlen * CFG::dmodel];
        float dense_encoder5[CFG::seqlen * CFG::dmodel];
        float dense_encoder6[CFG::seqlen * CFG::dmodel];
        float dense_encoder7[CFG::seqlen * CFG::dmodel];
        float dense_encoder8[CFG::seqlen * CFG::dmodel];
        float dense_encoder9[CFG::seqlen * CFG::dmodel];
        float dense_encoder10[CFG::seqlen * CFG::dmodel];
        float dense_encoder11[CFG::seqlen * CFG::dmodel];
        float dense_encoder12[CFG::seqlen * CFG::dmodel];

        /**
         * embedding load
         */
        load_from_txt<float, CFG::word * CFG::dmodel>(word_embeddings, "word_embedding_hls.txt");
        load_from_txt<float, CFG::pos * CFG::dmodel>(pos_embeddings, "pos_embedding_hls.txt");
        load_from_txt<float, CFG::token * CFG::dmodel>(token_embeddings, "token_embedding_hls.txt");
        load_from_txt<float, CFG::dmodel>(embed_norm_weight, "embed_norm_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(embed_norm_bias, "embed_norm_bias_hls.txt");

        /**
         * encoder 1 load
         */
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

        /**
         * encoder 2 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query1_weight_t, "encoder2_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key1_weight_t, "encoder2_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value1_weight_t, "encoder2_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense1_weight_t, "encoder2_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm1_weight, "encoder2_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query1_bias, "encoder2_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key1_bias, "encoder2_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value1_bias, "encoder2_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense1_bias, "encoder2_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm1_bias, "encoder2_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn1_dense_weight_t, "encoder2_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn1_dense_bias, "encoder2_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn1_dense2_weight_t, "encoder2_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_dense2_bias, "encoder2_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_norm_weight, "encoder2_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn1_norm_bias, "encoder2_ffn2_lnbias_hls.txt");

        /**
         * encoder 3 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query2_weight_t, "encoder3_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key2_weight_t, "encoder3_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value2_weight_t, "encoder3_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense2_weight_t, "encoder3_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm2_weight, "encoder3_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query2_bias, "encoder3_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key2_bias, "encoder3_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value2_bias, "encoder3_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense2_bias, "encoder3_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm2_bias, "encoder3_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn2_dense_weight_t, "encoder3_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn2_dense_bias, "encoder3_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn2_dense2_weight_t, "encoder3_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_dense2_bias, "encoder3_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_norm_weight, "encoder3_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn2_norm_bias, "encoder3_ffn2_lnbias_hls.txt");

        /**
         * encoder 4 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query3_weight_t, "encoder4_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key3_weight_t, "encoder4_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value3_weight_t, "encoder4_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense3_weight_t, "encoder4_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm3_weight, "encoder4_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query3_bias, "encoder4_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key3_bias, "encoder4_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value3_bias, "encoder4_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense3_bias, "encoder4_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm3_bias, "encoder4_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn3_dense_weight_t, "encoder4_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn3_dense_bias, "encoder4_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn3_dense2_weight_t, "encoder4_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_dense2_bias, "encoder4_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_norm_weight, "encoder4_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn3_norm_bias, "encoder4_ffn2_lnbias_hls.txt");

        /**
         * encoder 5 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query4_weight_t, "encoder5_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key4_weight_t, "encoder5_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value4_weight_t, "encoder5_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense4_weight_t, "encoder5_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm4_weight, "encoder5_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query4_bias, "encoder5_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key4_bias, "encoder5_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value4_bias, "encoder5_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense4_bias, "encoder5_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm4_bias, "encoder5_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn4_dense_weight_t, "encoder5_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn4_dense_bias, "encoder5_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn4_dense2_weight_t, "encoder5_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_dense2_bias, "encoder5_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_norm_weight, "encoder5_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn4_norm_bias, "encoder5_ffn2_lnbias_hls.txt");

        /**
         * encoder 6 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query5_weight_t, "encoder6_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key5_weight_t, "encoder6_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value5_weight_t, "encoder6_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense5_weight_t, "encoder6_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm5_weight, "encoder6_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query5_bias, "encoder6_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key5_bias, "encoder6_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value5_bias, "encoder6_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense5_bias, "encoder6_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm5_bias, "encoder6_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn5_dense_weight_t, "encoder6_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn5_dense_bias, "encoder6_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn5_dense2_weight_t, "encoder6_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_dense2_bias, "encoder6_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_norm_weight, "encoder6_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn5_norm_bias, "encoder6_ffn2_lnbias_hls.txt");

        /**
         * encoder 7 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query6_weight_t, "encoder7_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key6_weight_t, "encoder7_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value6_weight_t, "encoder7_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense6_weight_t, "encoder7_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm6_weight, "encoder7_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query6_bias, "encoder7_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key6_bias, "encoder7_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value6_bias, "encoder7_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense6_bias, "encoder7_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm6_bias, "encoder7_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn6_dense_weight_t, "encoder7_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn6_dense_bias, "encoder7_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn6_dense2_weight_t, "encoder7_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_dense2_bias, "encoder7_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_norm_weight, "encoder7_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn6_norm_bias, "encoder7_ffn2_lnbias_hls.txt");

        /**
         * encoder 8 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query7_weight_t, "encoder8_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key7_weight_t, "encoder8_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value7_weight_t, "encoder8_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense7_weight_t, "encoder8_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm7_weight, "encoder8_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query7_bias, "encoder8_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key7_bias, "encoder8_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value7_bias, "encoder8_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense7_bias, "encoder8_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm7_bias, "encoder8_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn7_dense_weight_t, "encoder8_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn7_dense_bias, "encoder8_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn7_dense2_weight_t, "encoder8_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_dense2_bias, "encoder8_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_norm_weight, "encoder8_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn7_norm_bias, "encoder8_ffn2_lnbias_hls.txt");

        /**
         * encoder 9 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query8_weight_t, "encoder9_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key8_weight_t, "encoder9_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value8_weight_t, "encoder9_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense8_weight_t, "encoder9_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm8_weight, "encoder9_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query8_bias, "encoder9_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key8_bias, "encoder9_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value8_bias, "encoder9_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense8_bias, "encoder9_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm8_bias, "encoder9_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn8_dense_weight_t, "encoder9_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn8_dense_bias, "encoder9_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn8_dense2_weight_t, "encoder9_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_dense2_bias, "encoder9_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_norm_weight, "encoder9_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn8_norm_bias, "encoder9_ffn2_lnbias_hls.txt");

        /**
         * encoder 10 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query9_weight_t, "encoder10_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key9_weight_t, "encoder10_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value9_weight_t, "encoder10_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense9_weight_t, "encoder10_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm9_weight, "encoder10_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query9_bias, "encoder10_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key9_bias, "encoder10_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value9_bias, "encoder10_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense9_bias, "encoder10_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm9_bias, "encoder10_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn9_dense_weight_t, "encoder10_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn9_dense_bias, "encoder10_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn9_dense2_weight_t, "encoder10_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_dense2_bias, "encoder10_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_norm_weight, "encoder10_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn9_norm_bias, "encoder10_ffn2_lnbias_hls.txt");

        /**
         * encoder 11 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query10_weight_t, "encoder11_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key10_weight_t, "encoder11_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value10_weight_t, "encoder11_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense10_weight_t, "encoder11_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm10_weight, "encoder11_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query10_bias, "encoder11_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key10_bias, "encoder11_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value10_bias, "encoder11_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense10_bias, "encoder11_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm10_bias, "encoder11_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn10_dense_weight_t, "encoder11_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn10_dense_bias, "encoder11_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn10_dense2_weight_t, "encoder11_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_dense2_bias, "encoder11_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_norm_weight, "encoder11_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn10_norm_bias, "encoder11_ffn2_lnbias_hls.txt");

        /**
         * encoder 12 load
         */
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(query11_weight_t, "encoder12_query_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(key11_weight_t, "encoder12_key_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(value11_weight_t, "encoder12_value_weight_hls.txt");
        load_from_txt<float, CFG::dmodel * CFG::dmodel>(dense11_weight_t, "encoder12_dense1_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm11_weight, "encoder12_lnorm_weight_hls.txt");

        load_from_txt<float, CFG::dmodel>(query11_bias, "encoder12_query_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(key11_bias, "encoder12_key_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(value11_bias, "encoder12_value_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(dense11_bias, "encoder12_dense1_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(norm11_bias, "encoder12_lnorm_bias_hls.txt");

        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn11_dense_weight_t, "encoder12_ffn1_weight_hls.txt");
        load_from_txt<float, CFG::ffdim>(ffn11_dense_bias, "encoder12_ffn1_bias_hls.txt");
        load_from_txt<float, CFG::ffdim * CFG::dmodel>(ffn11_dense2_weight_t, "encoder12_ffn2_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_dense2_bias, "encoder12_ffn2_bias_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_norm_weight, "encoder12_ffn2_lnweight_hls.txt");
        load_from_txt<float, CFG::dmodel>(ffn11_norm_bias, "encoder12_ffn2_lnbias_hls.txt");

        load_from_txt<float, CFG::dmodel * CFG::dmodel>(pooler_weight, "pooler_weight_hls.txt");
        load_from_txt<float, CFG::dmodel>(pooler_bias, "pooler_bias_hls.txt");

        embedding_out(word_embeddings, token_embeddings, pos_embeddings, input_ids, token_ids, pos_ids, embed_norm_weight, embed_norm_bias, embed_out);
        printf("embeded test = : \n");
        for (int i = 0; i < CFG::seqlen; ++i)
        {
                for (int j = 0; j < CFG::dmodel; ++j)
                {

                        printf("%f ", embed_out[i * CFG::dmodel + j]);
                }
                printf("\n");
        }

        encoder_kernel(embed_out, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias,
                       dense_weight_t, dense_bias, norm_weight, norm_bias, ffn_dense_weight_t, ffn_dense_bias, ffn_dense2_weight_t,
                       ffn_dense2_bias, ffn_norm_weight, ffn_norm_bias, dense_encoder1);

        encoder_kernel(dense_encoder1, query1_weight_t, query1_bias, key1_weight_t, key1_bias, value1_weight_t, value1_bias,
                       dense1_weight_t, dense1_bias, norm1_weight, norm1_bias, ffn1_dense_weight_t, ffn1_dense_bias, ffn1_dense2_weight_t,
                       ffn1_dense2_bias, ffn1_norm_weight, ffn1_norm_bias, dense_encoder2);

        encoder_kernel(dense_encoder2, query2_weight_t, query2_bias, key2_weight_t, key2_bias, value2_weight_t, value2_bias,
                       dense2_weight_t, dense2_bias, norm2_weight, norm2_bias, ffn2_dense_weight_t, ffn2_dense_bias, ffn2_dense2_weight_t,
                       ffn2_dense2_bias, ffn2_norm_weight, ffn2_norm_bias, dense_encoder3);

        encoder_kernel(dense_encoder3, query3_weight_t, query3_bias, key3_weight_t, key3_bias, value3_weight_t, value3_bias,
                       dense3_weight_t, dense3_bias, norm3_weight, norm3_bias, ffn3_dense_weight_t, ffn3_dense_bias, ffn3_dense2_weight_t,
                       ffn3_dense2_bias, ffn3_norm_weight, ffn3_norm_bias, dense_encoder4);

        encoder_kernel(dense_encoder4, query4_weight_t, query4_bias, key4_weight_t, key4_bias, value4_weight_t, value4_bias,
                       dense4_weight_t, dense4_bias, norm4_weight, norm4_bias, ffn4_dense_weight_t, ffn4_dense_bias, ffn4_dense2_weight_t,
                       ffn4_dense2_bias, ffn4_norm_weight, ffn4_norm_bias, dense_encoder5);

        encoder_kernel(dense_encoder5, query5_weight_t, query5_bias, key5_weight_t, key5_bias, value5_weight_t, value5_bias,
                       dense5_weight_t, dense5_bias, norm5_weight, norm5_bias, ffn5_dense_weight_t, ffn5_dense_bias, ffn5_dense2_weight_t,
                       ffn5_dense2_bias, ffn5_norm_weight, ffn5_norm_bias, dense_encoder6);

        encoder_kernel(dense_encoder6, query6_weight_t, query6_bias, key6_weight_t, key6_bias, value6_weight_t, value6_bias,
                       dense6_weight_t, dense6_bias, norm6_weight, norm6_bias, ffn6_dense_weight_t, ffn6_dense_bias, ffn6_dense2_weight_t,
                       ffn6_dense2_bias, ffn6_norm_weight, ffn6_norm_bias, dense_encoder7);

        encoder_kernel(dense_encoder7, query7_weight_t, query7_bias, key7_weight_t, key7_bias, value7_weight_t, value7_bias,
                       dense7_weight_t, dense7_bias, norm7_weight, norm7_bias, ffn7_dense_weight_t, ffn7_dense_bias, ffn7_dense2_weight_t,
                       ffn7_dense2_bias, ffn7_norm_weight, ffn7_norm_bias, dense_encoder8);

        encoder_kernel(dense_encoder8, query8_weight_t, query8_bias, key8_weight_t, key8_bias, value8_weight_t, value8_bias,
                       dense8_weight_t, dense8_bias, norm8_weight, norm8_bias, ffn8_dense_weight_t, ffn8_dense_bias, ffn8_dense2_weight_t,
                       ffn8_dense2_bias, ffn8_norm_weight, ffn8_norm_bias, dense_encoder9);

        encoder_kernel(dense_encoder9, query9_weight_t, query9_bias, key9_weight_t, key9_bias, value9_weight_t, value9_bias,
                       dense9_weight_t, dense9_bias, norm9_weight, norm9_bias, ffn9_dense_weight_t, ffn9_dense_bias, ffn9_dense2_weight_t,
                       ffn9_dense2_bias, ffn9_norm_weight, ffn9_norm_bias, dense_encoder10);

        encoder_kernel(dense_encoder10, query10_weight_t, query10_bias, key10_weight_t, key10_bias, value10_weight_t, value10_bias,
                       dense10_weight_t, dense10_bias, norm10_weight, norm10_bias, ffn10_dense_weight_t, ffn10_dense_bias, ffn10_dense2_weight_t,
                       ffn10_dense2_bias, ffn10_norm_weight, ffn10_norm_bias, dense_encoder11);

        encoder_kernel(dense_encoder11, query11_weight_t, query11_bias, key11_weight_t, key11_bias, value11_weight_t, value11_bias,
                       dense11_weight_t, dense11_bias, norm11_weight, norm11_bias, ffn11_dense_weight_t, ffn11_dense_bias, ffn11_dense2_weight_t,
                       ffn11_dense2_bias, ffn11_norm_weight, ffn11_norm_bias, dense_encoder12);

        linear_pooler(dense_encoder12, pooler_weight_T, pooler_bias, layer_out, 1, CFG::dmodel, CFG::dmodel);
}