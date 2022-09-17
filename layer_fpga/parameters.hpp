#include "config.hpp"
// #include "parameters_embedding.hpp"
/**
 * embedding
 */
float *hidden_states = new float[CFG::seqlen * CFG::dmodel];
float *word_embeddings = new float[CFG::word * CFG::dmodel];
float *pos_embeddings = new float[CFG::pos * CFG::dmodel];
float *token_embeddings = new float[CFG::token * CFG::dmodel];

float *pos_ids = new float[CFG::seqlen];
float *embed_norm_weight = new float[CFG::dmodel];
float *embed_norm_bias = new float[CFG::dmodel];
float *embed_out = new float[CFG::seqlen * CFG::dmodel];

/**
 * encoder 1
 */
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

/**
 * encoder 2
 */
float *query1_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key1_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value1_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query1_bias = new float[CFG::dmodel];
float *key1_bias = new float[CFG::dmodel];
float *value1_bias = new float[CFG::dmodel];
float *dense1_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense1_bias = new float[CFG::dmodel];
float *norm1_weight = new float[CFG::dmodel];
float *norm1_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn1_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn1_dense_bias = new float[CFG::ffdim];
float *ffn1_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn1_dense2_bias = new float[CFG::dmodel];
float *ffn1_norm_weight = new float[CFG::dmodel];
float *ffn1_norm_bias = new float[CFG::dmodel];

/**
 * encoder 3
 */
float *query2_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key2_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value2_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query2_bias = new float[CFG::dmodel];
float *key2_bias = new float[CFG::dmodel];
float *value2_bias = new float[CFG::dmodel];
float *dense2_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense2_bias = new float[CFG::dmodel];
float *norm2_weight = new float[CFG::dmodel];
float *norm2_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn2_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn2_dense_bias = new float[CFG::ffdim];
float *ffn2_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn2_dense2_bias = new float[CFG::dmodel];
float *ffn2_norm_weight = new float[CFG::dmodel];
float *ffn2_norm_bias = new float[CFG::dmodel];

/**
 * encoder 4
 */
float *query3_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key3_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value3_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query3_bias = new float[CFG::dmodel];
float *key3_bias = new float[CFG::dmodel];
float *value3_bias = new float[CFG::dmodel];
float *dense3_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense3_bias = new float[CFG::dmodel];
float *norm3_weight = new float[CFG::dmodel];
float *norm3_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn3_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn3_dense_bias = new float[CFG::ffdim];
float *ffn3_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn3_dense2_bias = new float[CFG::dmodel];
float *ffn3_norm_weight = new float[CFG::dmodel];
float *ffn3_norm_bias = new float[CFG::dmodel];

/**
 * encoder 5
 */
float *query4_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key4_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value4_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query4_bias = new float[CFG::dmodel];
float *key4_bias = new float[CFG::dmodel];
float *value4_bias = new float[CFG::dmodel];
float *dense4_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense4_bias = new float[CFG::dmodel];
float *norm4_weight = new float[CFG::dmodel];
float *norm4_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn4_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn4_dense_bias = new float[CFG::ffdim];
float *ffn4_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn4_dense2_bias = new float[CFG::dmodel];
float *ffn4_norm_weight = new float[CFG::dmodel];
float *ffn4_norm_bias = new float[CFG::dmodel];

/**
 * encoder 6
 */
float *query5_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key5_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value5_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query5_bias = new float[CFG::dmodel];
float *key5_bias = new float[CFG::dmodel];
float *value5_bias = new float[CFG::dmodel];
float *dense5_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense5_bias = new float[CFG::dmodel];
float *norm5_weight = new float[CFG::dmodel];
float *norm5_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn5_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn5_dense_bias = new float[CFG::ffdim];
float *ffn5_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn5_dense2_bias = new float[CFG::dmodel];
float *ffn5_norm_weight = new float[CFG::dmodel];
float *ffn5_norm_bias = new float[CFG::dmodel];

/**
 * encoder 7
 */
float *query6_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key6_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value6_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query6_bias = new float[CFG::dmodel];
float *key6_bias = new float[CFG::dmodel];
float *value6_bias = new float[CFG::dmodel];
float *dense6_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense6_bias = new float[CFG::dmodel];
float *norm6_weight = new float[CFG::dmodel];
float *norm6_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn6_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn6_dense_bias = new float[CFG::ffdim];
float *ffn6_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn6_dense2_bias = new float[CFG::dmodel];
float *ffn6_norm_weight = new float[CFG::dmodel];
float *ffn6_norm_bias = new float[CFG::dmodel];

/**
 * encoder 8
 */
float *query7_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key7_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value7_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query7_bias = new float[CFG::dmodel];
float *key7_bias = new float[CFG::dmodel];
float *value7_bias = new float[CFG::dmodel];
float *dense7_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense7_bias = new float[CFG::dmodel];
float *norm7_weight = new float[CFG::dmodel];
float *norm7_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn7_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn7_dense_bias = new float[CFG::ffdim];
float *ffn7_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn7_dense2_bias = new float[CFG::dmodel];
float *ffn7_norm_weight = new float[CFG::dmodel];
float *ffn7_norm_bias = new float[CFG::dmodel];

/**
 * encoder 9
 */
float *query8_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key8_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value8_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query8_bias = new float[CFG::dmodel];
float *key8_bias = new float[CFG::dmodel];
float *value8_bias = new float[CFG::dmodel];
float *dense8_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense8_bias = new float[CFG::dmodel];
float *norm8_weight = new float[CFG::dmodel];
float *norm8_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn8_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn8_dense_bias = new float[CFG::ffdim];
float *ffn8_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn8_dense2_bias = new float[CFG::dmodel];
float *ffn8_norm_weight = new float[CFG::dmodel];
float *ffn8_norm_bias = new float[CFG::dmodel];

/**
 * encoder 10
 */
float *query9_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key9_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value9_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query9_bias = new float[CFG::dmodel];
float *key9_bias = new float[CFG::dmodel];
float *value9_bias = new float[CFG::dmodel];
float *dense9_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense9_bias = new float[CFG::dmodel];
float *norm9_weight = new float[CFG::dmodel];
float *norm9_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn9_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn9_dense_bias = new float[CFG::ffdim];
float *ffn9_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn9_dense2_bias = new float[CFG::dmodel];
float *ffn9_norm_weight = new float[CFG::dmodel];
float *ffn9_norm_bias = new float[CFG::dmodel];

/**
 * encoder 11
 */
float *query10_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key10_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value10_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query10_bias = new float[CFG::dmodel];
float *key10_bias = new float[CFG::dmodel];
float *value10_bias = new float[CFG::dmodel];
float *dense10_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense10_bias = new float[CFG::dmodel];
float *norm10_weight = new float[CFG::dmodel];
float *norm10_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn10_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn10_dense_bias = new float[CFG::ffdim];
float *ffn10_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn10_dense2_bias = new float[CFG::dmodel];
float *ffn10_norm_weight = new float[CFG::dmodel];
float *ffn10_norm_bias = new float[CFG::dmodel];

/**
 * encoder 12
 */
float *query11_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *key11_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *value11_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *query11_bias = new float[CFG::dmodel];
float *key11_bias = new float[CFG::dmodel];
float *value11_bias = new float[CFG::dmodel];
float *dense11_weight_t = new float[CFG::dmodel * CFG::dmodel];
float *dense11_bias = new float[CFG::dmodel];
float *norm11_weight = new float[CFG::dmodel];
float *norm11_bias = new float[CFG::dmodel];

/*ffn parameters*/
float *ffn11_dense_weight_t = new float[CFG::ffdim * CFG::dmodel];
float *ffn11_dense_bias = new float[CFG::ffdim];
float *ffn11_dense2_weight_t = new float[CFG::dmodel * CFG::ffdim];
float *ffn11_dense2_bias = new float[CFG::dmodel];
float *ffn11_norm_weight = new float[CFG::dmodel];
float *ffn11_norm_bias = new float[CFG::dmodel];

float *pooler_weight = new float[CFG::dmodel * CFG::dmodel];
float *pooler_bias = new float[CFG::dmodel];
float *pooler_weight_T = new float[CFG::dmodel * CFG::dmodel];