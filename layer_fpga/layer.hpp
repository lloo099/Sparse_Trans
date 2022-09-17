
void embedding_lut(float *in, float *out, float *index, const int A, const int B);

void matrix2d_transpose(float *in, float *out, const int A, const int B);

void layernorm_base(float *act, float *y, float *norm_weight, float *norm_bias);

void linear_pooler(float *A, float *B, float *bias, float *out, const int OH, const int OW, const int K);

void embedding_out(float *word_embeddings, float *token_embeddings, float *pos_embeddings, float *input_ids, float *token_ids, float *pos_ids, float *embed_norm_weight, float *embed_norm_bias, float *dense_out);

void attention_layer(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                     float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *out);

void attention_kernel(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                      float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *out);

void FFN_kernel(float *fc_in, float *dense_weight_t, float *dense_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out);

void qkv(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
         float *query_in, float *key_in, float *value_in);

void FFN(float *fc_in, float *dense_weight_t, float *dense_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out);

void encoder(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
             float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *dense1_weight_t,
             float *dense1_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out);

void encoder_kernel(float *in, float *query_weight_t, float *query_bias, float *key_weight_t, float *key_bias, float *value_weight_t, float *value_bias,
                    float *dense_weight_t, float *dense_bias, float *norm_weight, float *norm_bias, float *dense1_weight_t,
                    float *dense1_bias, float *dense2_weight_t, float *dense2_bias, float *ffn_norm_weight, float *ffn_norm_bias, float *dense_out);
