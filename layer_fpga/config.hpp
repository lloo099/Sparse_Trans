#pragma once

//  BERT config:
// "attention_probs_dropout_prob" : 0.1,
//     "classifier_dropout" : null,
//     "gradient_checkpointing" : false,
//      "hidden_act" : "gelu",
//      "hidden_dropout_prob" : 0.1,
//      "hidden_size" : 768,
//      "initializer_range" : 0.02,
//      "intermediate_size" : 3072,
//      "layer_norm_eps" : 1e-12,
//      "max_position_embeddings" : 512,
//      "model_type" : "bert",
//      "num_attention_heads" : 12,
//      "num_hidden_layers" : 12,
//      "pad_token_id" : 0,
//      "position_embedding_type" : "absolute",
//       "transformers_version" : "4.20.1",
//       "type_vocab_size" : 2,
//        "use_cache" : true,
//        "vocab_size" : 30522

extern "C"
{
    namespace CFG
    {
        const int seqlen = 128;
        const int nhead = 12;
        const int dhead = 64;
        const int dmodel = 768;
        const int ffdim = 3072;
        const int word = 30522;
        const int token = 2;
        const int pos = 512;
        const float eps = 1e-5;

    } // namespace CONFIG
}
