from distutils.command.config import config
import torch
import torch.nn as nn
import math
from transformers import BertModel, BertTokenizer
import os
import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset

from json import JSONEncoder
import json


def attention(layer, hidden_states, attention_mask=None):
    '''
    Pass in a encoder layer (which holds pretrained weights) and hidden_states input,
    and this function performs the same operations as the layer but in a readable fashion.

    hidden_states: <bs, seqlen, dmodel>
    '''
    bs, seqlen, dmodel = hidden_states.size()
    num_heads = layer.attention.self.num_attention_heads
    dhead = layer.attention.self.attention_head_size

    # Linear transform to get multiple heads. This is a major MAC slurper.
    # Each of these is calling an nn.Linear layer on hidden_states.
#     query_layer = layer.attention.self.query(hidden_states) # <bs, seqlen, dmodel>
    query_layer = layer.attention.self.query(hidden_states)
    key_layer = layer.attention.self.key(
        hidden_states)     # <bs, seqlen, dmodel>
    value_layer = layer.attention.self.value(
        hidden_states)  # <bs, seqlen, dmodel>


# save for c++ check
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_q_layer_weight.txt'),
               transform_feature(layer.attention.self.query.weight.detach().numpy()), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_q_layer_out.txt'),
               transform_feature(query_layer.detach().numpy()), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_k_layer_weight.txt'),
               transform_feature(layer.attention.self.key.weight.detach().numpy()), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_k_layer_out.txt'),
               transform_feature(key_layer.detach().numpy()), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_v_layer_weight.txt'),
               transform_feature(layer.attention.self.value.weight.detach().numpy()), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_v_layer_out.txt'),
               transform_feature(value_layer.detach().numpy()), fmt='%f')
    print("Q : ", query_layer.shape)
    print("K : ", key_layer.shape)
    print("V : ", value_layer.shape)

    # Reshape and transpose for multi-head
    new_shape = (bs, seqlen, num_heads, dhead)
    print("new shape : ", new_shape)
    query_layer = query_layer.view(new_shape)
    value_layer = value_layer.view(new_shape)
    key_layer = key_layer.view(new_shape)

    # <bs, num_head, seqlen, dhead>
    query_layer = query_layer.permute(0, 2, 1, 3)
    # <bs, num_head, seqlen, dhead>
    value_layer = value_layer.permute(0, 2, 1, 3)
    # Key is transposed to match dimensions of Query for matmul
    # <bs, num_head, dhead, seqlen>
    key_layer = key_layer.permute(0, 2, 3, 1)

    # The attention main course
    attention_scores = torch.matmul(query_layer, key_layer)
    print("attention score : ", attention_scores.shape)
    attention_scores /= math.sqrt(dhead)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_attentionscores_layer_out.txt'),
               transform_feature(attention_scores.detach().numpy()), fmt='%f')
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
    #     attention_scores = attention_scores + attention_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    # attention_probs = layer.attention.self.dropout(attention_probs)

    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_soft_layer_out.txt'),
               transform_feature(attention_probs.detach().numpy()), fmt='%f')
    # Weighted sum of Values from softmax attention
    attention_out = torch.matmul(attention_probs, value_layer)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_value_attention_layer_out.txt'),
               transform_feature(attention_out.detach().numpy()), fmt='%f')
    attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
    attention_out = attention_out.view(bs, seqlen, dmodel)

    # It's time for one more linear transform and layer norm
    dense_out = layer.attention.output.dense(attention_out)
    print("dense out before dropout: ", dense_out.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_dense1_layer_before_dropout.txt'),
               transform_feature(dense_out.detach().numpy()), fmt='%f')
    dense_out = layer.attention.output.dropout(dense_out)
    print("dense out after dropout: ", dense_out.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_dense1_layer_after_dropout.txt'),
               transform_feature(dense_out.detach().numpy()), fmt='%f')
    # LayerNorm also mplements the residual connection
    dense_out = dense_out + hidden_states
    print("dense out : ", dense_out.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_dense1_layer_out.txt'),
               transform_feature(dense_out.detach().numpy()), fmt='%f')
    layer_out = layer.attention.output.LayerNorm(
        dense_out)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_layernore_layer_out.txt'),
               transform_feature(layer_out.detach().numpy()), fmt='%f')
    return layer_out


def ffn(layer, attention_out):
    '''
    Pass in the feedforward layer and attention output. Returns the same result of a FF forward pass.
    '''
    # Layer 1
    output = layer.intermediate.dense(attention_out)
    print("ffn1 out : ", output.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn1_layer_out.txt'),
               transform_feature(output.detach().numpy()), fmt='%f')

    output = nn.functional.gelu(output)
    print("gelu out : ", output.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_gelu_layer_out.txt'),
               transform_feature(output.detach().numpy()), fmt='%f')
    # Layer 2
    output = layer.output.dense(output)
    print("ffn2 out : ", output.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_layer_out.txt'),
               transform_feature(output.detach().numpy()), fmt='%f')

    output = layer.output.dropout(output)
    output = layer.output.LayerNorm(output + attention_out)

    print("ffn2 ln out : ", output.shape)

    return output


def encoder(model, hidden_states, attention_mask):
    '''
    Pass input through encoder stack
    '''
    for layer_module in model.encoder.layer:
        # MHA + LayerNorm
        attention_out = attention(layer_module, hidden_states, attention_mask)
        ff_out = ffn(layer_module, attention_out)
        hidden_states = ff_out

    return hidden_states


def transform_w(feature):
    feature = np.array(feature)
    feature = np.squeeze(feature)
    feature = np.transpose(feature, (1, 0))
    feature = np.ndarray.flatten(feature)
    return feature


def transform_feature(feature):
    feature = np.array(feature)
    feature = np.squeeze(feature)
    # feature = np.transpose(feature, (1,2,3,0))
    # feature = np.transpose(feature, (0, 3, 2, 1))
    feature = np.ndarray.flatten(feature)
    return feature


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print('\n\n, Baseline Bert modules: \n', model)

    # Input is the first 512 tokens generated from the proposal for this project.
    text_512 = 'This project aims to implement a transformer layer on a cluster of FPGAs. In recent years transformers have outperformed traditional convolutional neural networks in many fields, but serial performance is dismal and parallel GPU performance is power-intensive. Specialized architectures have been studied little, especially using FPGA platforms. This research will improve transformer inference performance by offloading computationally intensive sections of the network to reconfigurable accelerators running on a cluster of multiple FPGA devices. This research will result in an acceleration architecture for a single layer of a transformer network along with a performance comparison with CPU and GPU baselines. We propose the investigation of distributed transformer inference across a cluster of multiple field programmable gate arrays (FPGAs). This research will investigate the partitioning of a transformer layer across multiple FPGA devices along with networking between FPGAs in the cluster. Transformers have become a dominant machine learning architecture for many domains such as natural language processing, therefore high speed inference is desirable. However, networks sizes and limited FPGA resources often make inference on a single FPGA slow due to limited parallelism and pipeline depth or impossible due to limited resources. The purpose of this research is to explore methods to overcome these challenges by introducing parallelism through multi-FPGA clusters. Transformers are highly parallel neural network architectures which consist of stacks of encoder and decoder layers. These layers consist of many linear transformations on matrices which are represented by matrix-matrix multiplication. Within an encoder/decoder layer there is an opportunity to parallelize both between concurrent general matrix multiplies (GeMM) and within each GeMM. Attempting to serialize these operations on a CPU leads to high execution time and is a poor utilization of the CPU\'s general purpose architecture. GPUs can deliver high throughput inference for transformers, though they are power-hungry and do not achieve the low latency required by some applications. Both in the datacenter and at the edge, low-latency and efficient inference is desired. Optimally, there would be an architecture that could scale between these two extremes of computational demand. State-of-the-art transformers can contain upwards of 12 layers and multiply matrices on the order of 1024x1024 elements. In addition, the trend of increasing transformer size does not show signs of slowing. This large use of memory and FLOPs leads to difficulty mapping an entire transformer network to a '
    text_128 = 'This project aims to implement a transformer layer on a cluster of FPGAs. In recent years transformers have outperformed traditional convolutional neural networks in many fields, but serial performance is dismal and parallel GPU performance is power-intensive. Specialized architectures have been studied little, especially using FPGA platforms. This research will improve transformer inference performance by offloading computationally intensive sections of the network to reconfigurable accelerators running on a cluster of multiple FPGA devices. This research will result in an acceleration architecture for a single layer of a transformer network along with a  '
    text = text_128
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    output.last_hidden_state.shape, output.last_hidden_state

    embedding_output = model.embeddings(
        input_ids=encoded_input['input_ids'],
        position_ids=None,
        token_type_ids=encoded_input['token_type_ids'],
        inputs_embeds=None,
        past_key_values_length=0,
    )

    txt_dir = "/home/enai/Downloads/trans-fat/src/baseline/layer/tb_pytorch/layer_parameters/"
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    hidden_states = embedding_output
    np.savetxt(os.path.join(txt_dir, txt_dir + 'embedding_in.txt'),
               transform_feature(hidden_states.detach().numpy()), fmt='%f')

    embedding = model.embeddings
    torch.set_printoptions(profile="default")
    wordembedding = embedding.state_dict()['word_embeddings.weight']
    tokenembedding = embedding.state_dict()['token_type_embeddings.weight']
    posembedding = embedding.state_dict()['position_embeddings.weight']
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']

    embed_norm_weight = embedding.state_dict()['LayerNorm.weight']
    embed_norm_bias = embedding.state_dict()['LayerNorm.bias']
    # print(position_ids)
    # config = model.config
    # print("config :", config)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'word_embedding.txt'),
               transform_feature(wordembedding), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'token_embedding.txt'),
               transform_feature(tokenembedding), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'pos_embedding.txt'),
               transform_feature(posembedding), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoded_input.txt'),
               transform_feature(input_ids), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoded_token.txt'),
               transform_feature(token_type_ids), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'embed_norm_weight.txt'),
               transform_feature(embed_norm_weight), fmt='%f')
    np.savetxt(os.path.join(txt_dir, txt_dir + 'embed_norm_bias.txt'),
               transform_feature(embed_norm_bias), fmt='%f')
    # np.savetxt(os.path.join(txt_dir, txt_dir + 'encoded_pos.txt'),
    #            transform_feature(position_ids), fmt='%f')
    print("input ids: ", input_ids.shape)
    print("token type ids: ", token_type_ids.shape)
    # print("pos ids: ", position_ids)
    print("word embedding: ", wordembedding.shape)
    print("token embedding: ", tokenembedding.shape)
    print("pos embedding: ", posembedding.shape)
    print("embedding norm weight: ", embed_norm_weight.shape)
    print("embedding norm bias: ", embed_norm_bias.shape)
    print("hidden_states: ", hidden_states.shape)
    # for layer_module in model.encoder.layer:
    #   # MHA + LayerNorm
    #     attention_out = attention(layer_module, hidden_states)
    #     ff_out = ffn(layer_module, attention_out)
    #     hidden_states = ff_out
    layer = model.encoder.layer[0]
    torch.save(model.state_dict(), 'save.pt')
    attention_out = attention(layer, hidden_states)

    print("attention out: ", attention_out.shape)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_attention_out.txt'),
               transform_feature(attention_out.detach().numpy()), fmt='%f')
    ff_out = ffn(layer, attention_out)
    np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_ln_layer_out.txt'),
               transform_feature(ff_out.detach().numpy()), fmt='%f')

    # layer = model.encoder.layer[1]

    # attention_out = attention(layer, ff_out)

    # print("attention out: ", attention_out.shape)
    # np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_attention_out.txt'),
    #            transform_feature(attention_out.detach().numpy()), fmt='%f')
    # ff_out = ffn(layer, attention_out)
    # np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn2_ln_layer_out.txt'),
    #            transform_feature(ff_out.detach().numpy()), fmt='%f')
    # torch.set_printoptions(profile="default")
    # print(layer)

#     # weight+bias save - encoder 1
#     encoder1_query_weight = layer.state_dict()['attention.self.query.weight']
#     encoder1_query_bias = layer.state_dict()['attention.self.query.bias']
#     encoder1_key_weight = layer.state_dict()['attention.self.key.weight']
#     encoder1_key_bias = layer.state_dict()['attention.self.key.bias']
#     encoder1_value_weight = layer.state_dict()['attention.self.value.weight']
#     encoder1_value_bias = layer.state_dict()['attention.self.value.bias']
#     encoder1_dense1_weight = layer.state_dict(
#     )['attention.output.dense.weight']
#     encoder1_dense1_bias = layer.state_dict()['attention.output.dense.bias']
#     encoder1_lnorm_weight = layer.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder1_lnorm_bias = layer.state_dict()['attention.output.LayerNorm.bias']
#     print("Q weight: ", encoder1_query_weight.shape)
#     print("Q bias: ", encoder1_query_bias.shape)
#     print("LN weight: ", encoder1_lnorm_weight.shape)
#     print("LN Bias: ", encoder1_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_query_weight.txt'),
#                transform_w(encoder1_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_query_bias.txt'),
#                encoder1_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_key_weight.txt'),
#                transform_w(encoder1_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_key_bias.txt'),
#                encoder1_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_value_weight.txt'),
#                transform_w(encoder1_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_value_bias.txt'),
#                encoder1_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_dense1_weight.txt'),
#                transform_w(encoder1_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_dense1_bias.txt'),
#                encoder1_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_lnorm_weight.txt'),
#                transform_feature(encoder1_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_lnorm_bias.txt'),
#                encoder1_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer.state_dict()['output.dense.weight']
#     ffn2_b = layer.state_dict()['output.dense.bias']
#     ffn2_lnw = layer.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder1_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 2
#     layer2 = model.encoder.layer[1]
#     encoder2_query_weight = layer2.state_dict()['attention.self.query.weight']
#     encoder2_query_bias = layer2.state_dict()['attention.self.query.bias']
#     encoder2_key_weight = layer2.state_dict()['attention.self.key.weight']
#     encoder2_key_bias = layer2.state_dict()['attention.self.key.bias']
#     encoder2_value_weight = layer2.state_dict()['attention.self.value.weight']
#     encoder2_value_bias = layer2.state_dict()['attention.self.value.bias']
#     encoder2_dense1_weight = layer2.state_dict(
#     )['attention.output.dense.weight']
#     encoder2_dense1_bias = layer2.state_dict()['attention.output.dense.bias']
#     encoder2_lnorm_weight = layer2.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder2_lnorm_bias = layer2.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q2 weight: ", encoder2_query_weight.shape)
#     print("Q2 bias: ", encoder2_query_bias.shape)
#     print("LN2 weight: ", encoder2_lnorm_weight.shape)
#     print("LN2 Bias: ", encoder2_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_query_weight.txt'),
#                transform_w(encoder2_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_query_bias.txt'),
#                encoder2_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_key_weight.txt'),
#                transform_w(encoder2_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_key_bias.txt'),
#                encoder2_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_value_weight.txt'),
#                transform_w(encoder2_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_value_bias.txt'),
#                encoder2_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_dense1_weight.txt'),
#                transform_w(encoder2_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_dense1_bias.txt'),
#                encoder2_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_lnorm_weight.txt'),
#                transform_feature(encoder2_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_lnorm_bias.txt'),
#                encoder2_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer2.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer2.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer2.state_dict()['output.dense.weight']
#     ffn2_b = layer2.state_dict()['output.dense.bias']
#     ffn2_lnw = layer2.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer2.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder2_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 3
#     layer3 = model.encoder.layer[2]
#     encoder3_query_weight = layer3.state_dict()['attention.self.query.weight']
#     encoder3_query_bias = layer3.state_dict()['attention.self.query.bias']
#     encoder3_key_weight = layer3.state_dict()['attention.self.key.weight']
#     encoder3_key_bias = layer3.state_dict()['attention.self.key.bias']
#     encoder3_value_weight = layer3.state_dict()['attention.self.value.weight']
#     encoder3_value_bias = layer3.state_dict()['attention.self.value.bias']
#     encoder3_dense1_weight = layer3.state_dict(
#     )['attention.output.dense.weight']
#     encoder3_dense1_bias = layer3.state_dict()['attention.output.dense.bias']
#     encoder3_lnorm_weight = layer3.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder3_lnorm_bias = layer3.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q3 weight: ", encoder3_query_weight.shape)
#     print("Q3 bias: ", encoder3_query_bias.shape)
#     print("LN3 weight: ", encoder3_lnorm_weight.shape)
#     print("LN3 Bias: ", encoder3_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_query_weight.txt'),
#                transform_w(encoder3_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_query_bias.txt'),
#                encoder3_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_key_weight.txt'),
#                transform_w(encoder3_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_key_bias.txt'),
#                encoder3_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_value_weight.txt'),
#                transform_w(encoder3_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_value_bias.txt'),
#                encoder3_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_dense1_weight.txt'),
#                transform_w(encoder3_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_dense1_bias.txt'),
#                encoder3_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_lnorm_weight.txt'),
#                transform_feature(encoder3_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_lnorm_bias.txt'),
#                encoder3_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer3.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer3.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer3.state_dict()['output.dense.weight']
#     ffn2_b = layer3.state_dict()['output.dense.bias']
#     ffn2_lnw = layer3.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer3.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder3_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 4
#     layer4 = model.encoder.layer[3]
#     encoder4_query_weight = layer4.state_dict()['attention.self.query.weight']
#     encoder4_query_bias = layer4.state_dict()['attention.self.query.bias']
#     encoder4_key_weight = layer4.state_dict()['attention.self.key.weight']
#     encoder4_key_bias = layer4.state_dict()['attention.self.key.bias']
#     encoder4_value_weight = layer4.state_dict()['attention.self.value.weight']
#     encoder4_value_bias = layer4.state_dict()['attention.self.value.bias']
#     encoder4_dense1_weight = layer4.state_dict(
#     )['attention.output.dense.weight']
#     encoder4_dense1_bias = layer4.state_dict()['attention.output.dense.bias']
#     encoder4_lnorm_weight = layer4.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder4_lnorm_bias = layer4.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q4 weight: ", encoder4_query_weight.shape)
#     print("Q4 bias: ", encoder4_query_bias.shape)
#     print("LN4 weight: ", encoder4_lnorm_weight.shape)
#     print("LN4 Bias: ", encoder4_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_query_weight.txt'),
#                transform_w(encoder4_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_query_bias.txt'),
#                encoder4_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_key_weight.txt'),
#                transform_w(encoder4_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_key_bias.txt'),
#                encoder4_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_value_weight.txt'),
#                transform_w(encoder4_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_value_bias.txt'),
#                encoder4_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_dense1_weight.txt'),
#                transform_w(encoder4_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_dense1_bias.txt'),
#                encoder4_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_lnorm_weight.txt'),
#                transform_feature(encoder4_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_lnorm_bias.txt'),
#                encoder4_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer4.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer4.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer4.state_dict()['output.dense.weight']
#     ffn2_b = layer4.state_dict()['output.dense.bias']
#     ffn2_lnw = layer4.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer4.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder4_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 5
#     layer5 = model.encoder.layer[4]
#     encoder5_query_weight = layer5.state_dict()['attention.self.query.weight']
#     encoder5_query_bias = layer5.state_dict()['attention.self.query.bias']
#     encoder5_key_weight = layer5.state_dict()['attention.self.key.weight']
#     encoder5_key_bias = layer5.state_dict()['attention.self.key.bias']
#     encoder5_value_weight = layer5.state_dict()['attention.self.value.weight']
#     encoder5_value_bias = layer5.state_dict()['attention.self.value.bias']
#     encoder5_dense1_weight = layer5.state_dict(
#     )['attention.output.dense.weight']
#     encoder5_dense1_bias = layer5.state_dict()['attention.output.dense.bias']
#     encoder5_lnorm_weight = layer5.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder5_lnorm_bias = layer5.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q5 weight: ", encoder5_query_weight.shape)
#     print("Q5 bias: ", encoder5_query_bias.shape)
#     print("LN5 weight: ", encoder5_lnorm_weight.shape)
#     print("LN5 Bias: ", encoder5_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_query_weight.txt'),
#                transform_w(encoder5_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_query_bias.txt'),
#                encoder5_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_key_weight.txt'),
#                transform_w(encoder5_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_key_bias.txt'),
#                encoder5_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_value_weight.txt'),
#                transform_w(encoder5_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_value_bias.txt'),
#                encoder5_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_dense1_weight.txt'),
#                transform_w(encoder5_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_dense1_bias.txt'),
#                encoder5_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_lnorm_weight.txt'),
#                transform_feature(encoder5_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_lnorm_bias.txt'),
#                encoder5_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer5.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer5.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer5.state_dict()['output.dense.weight']
#     ffn2_b = layer5.state_dict()['output.dense.bias']
#     ffn2_lnw = layer5.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer5.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder5_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 6
#     layer6 = model.encoder.layer[5]
#     encoder6_query_weight = layer6.state_dict()['attention.self.query.weight']
#     encoder6_query_bias = layer6.state_dict()['attention.self.query.bias']
#     encoder6_key_weight = layer6.state_dict()['attention.self.key.weight']
#     encoder6_key_bias = layer6.state_dict()['attention.self.key.bias']
#     encoder6_value_weight = layer6.state_dict()['attention.self.value.weight']
#     encoder6_value_bias = layer6.state_dict()['attention.self.value.bias']
#     encoder6_dense1_weight = layer6.state_dict(
#     )['attention.output.dense.weight']
#     encoder6_dense1_bias = layer6.state_dict()['attention.output.dense.bias']
#     encoder6_lnorm_weight = layer6.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder6_lnorm_bias = layer6.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q6 weight: ", encoder6_query_weight.shape)
#     print("Q6 bias: ", encoder6_query_bias.shape)
#     print("LN6 weight: ", encoder6_lnorm_weight.shape)
#     print("LN6 Bias: ", encoder6_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_query_weight.txt'),
#                transform_w(encoder6_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_query_bias.txt'),
#                encoder6_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_key_weight.txt'),
#                transform_w(encoder6_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_key_bias.txt'),
#                encoder6_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_value_weight.txt'),
#                transform_w(encoder6_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_value_bias.txt'),
#                encoder6_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_dense1_weight.txt'),
#                transform_w(encoder6_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_dense1_bias.txt'),
#                encoder6_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_lnorm_weight.txt'),
#                transform_feature(encoder6_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_lnorm_bias.txt'),
#                encoder6_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer6.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer6.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer6.state_dict()['output.dense.weight']
#     ffn2_b = layer6.state_dict()['output.dense.bias']
#     ffn2_lnw = layer6.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer6.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder6_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 7
#     layer7 = model.encoder.layer[6]
#     encoder7_query_weight = layer7.state_dict()['attention.self.query.weight']
#     encoder7_query_bias = layer7.state_dict()['attention.self.query.bias']
#     encoder7_key_weight = layer7.state_dict()['attention.self.key.weight']
#     encoder7_key_bias = layer7.state_dict()['attention.self.key.bias']
#     encoder7_value_weight = layer7.state_dict()['attention.self.value.weight']
#     encoder7_value_bias = layer7.state_dict()['attention.self.value.bias']
#     encoder7_dense1_weight = layer7.state_dict(
#     )['attention.output.dense.weight']
#     encoder7_dense1_bias = layer7.state_dict()['attention.output.dense.bias']
#     encoder7_lnorm_weight = layer7.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder7_lnorm_bias = layer7.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q7 weight: ", encoder7_query_weight.shape)
#     print("Q7 bias: ", encoder7_query_bias.shape)
#     print("LN7 weight: ", encoder7_lnorm_weight.shape)
#     print("LN7 Bias: ", encoder7_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_query_weight.txt'),
#                transform_w(encoder7_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_query_bias.txt'),
#                encoder7_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_key_weight.txt'),
#                transform_w(encoder7_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_key_bias.txt'),
#                encoder7_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_value_weight.txt'),
#                transform_w(encoder7_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_value_bias.txt'),
#                encoder7_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_dense1_weight.txt'),
#                transform_w(encoder7_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_dense1_bias.txt'),
#                encoder7_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_lnorm_weight.txt'),
#                transform_feature(encoder7_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_lnorm_bias.txt'),
#                encoder7_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer7.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer7.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer7.state_dict()['output.dense.weight']
#     ffn2_b = layer7.state_dict()['output.dense.bias']
#     ffn2_lnw = layer7.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer7.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder7_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 8
#     layer8 = model.encoder.layer[7]
#     encoder8_query_weight = layer8.state_dict()['attention.self.query.weight']
#     encoder8_query_bias = layer8.state_dict()['attention.self.query.bias']
#     encoder8_key_weight = layer8.state_dict()['attention.self.key.weight']
#     encoder8_key_bias = layer8.state_dict()['attention.self.key.bias']
#     encoder8_value_weight = layer8.state_dict()['attention.self.value.weight']
#     encoder8_value_bias = layer8.state_dict()['attention.self.value.bias']
#     encoder8_dense1_weight = layer8.state_dict(
#     )['attention.output.dense.weight']
#     encoder8_dense1_bias = layer8.state_dict()['attention.output.dense.bias']
#     encoder8_lnorm_weight = layer8.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder8_lnorm_bias = layer8.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q8 weight: ", encoder8_query_weight.shape)
#     print("Q8 bias: ", encoder8_query_bias.shape)
#     print("LN8 weight: ", encoder8_lnorm_weight.shape)
#     print("LN8 Bias: ", encoder8_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_query_weight.txt'),
#                transform_w(encoder8_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_query_bias.txt'),
#                encoder8_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_key_weight.txt'),
#                transform_w(encoder8_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_key_bias.txt'),
#                encoder8_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_value_weight.txt'),
#                transform_w(encoder8_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_value_bias.txt'),
#                encoder8_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_dense1_weight.txt'),
#                transform_w(encoder8_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_dense1_bias.txt'),
#                encoder8_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_lnorm_weight.txt'),
#                transform_feature(encoder8_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_lnorm_bias.txt'),
#                encoder8_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer8.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer8.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer8.state_dict()['output.dense.weight']
#     ffn2_b = layer8.state_dict()['output.dense.bias']
#     ffn2_lnw = layer8.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer8.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder8_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 5
#     layer9 = model.encoder.layer[8]
#     encoder9_query_weight = layer9.state_dict()['attention.self.query.weight']
#     encoder9_query_bias = layer9.state_dict()['attention.self.query.bias']
#     encoder9_key_weight = layer9.state_dict()['attention.self.key.weight']
#     encoder9_key_bias = layer9.state_dict()['attention.self.key.bias']
#     encoder9_value_weight = layer9.state_dict()['attention.self.value.weight']
#     encoder9_value_bias = layer9.state_dict()['attention.self.value.bias']
#     encoder9_dense1_weight = layer9.state_dict(
#     )['attention.output.dense.weight']
#     encoder9_dense1_bias = layer9.state_dict()['attention.output.dense.bias']
#     encoder9_lnorm_weight = layer9.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder9_lnorm_bias = layer9.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q9 weight: ", encoder9_query_weight.shape)
#     print("Q9 bias: ", encoder9_query_bias.shape)
#     print("LN9 weight: ", encoder9_lnorm_weight.shape)
#     print("LN9 Bias: ", encoder9_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_query_weight.txt'),
#                transform_w(encoder9_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_query_bias.txt'),
#                encoder9_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_key_weight.txt'),
#                transform_w(encoder9_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_key_bias.txt'),
#                encoder9_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_value_weight.txt'),
#                transform_w(encoder9_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_value_bias.txt'),
#                encoder9_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_dense1_weight.txt'),
#                transform_w(encoder9_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_dense1_bias.txt'),
#                encoder9_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_lnorm_weight.txt'),
#                transform_feature(encoder9_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_lnorm_bias.txt'),
#                encoder9_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer9.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer9.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer9.state_dict()['output.dense.weight']
#     ffn2_b = layer9.state_dict()['output.dense.bias']
#     ffn2_lnw = layer9.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer9.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder9_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 10
#     layer10 = model.encoder.layer[9]
#     encoder10_query_weight = layer10.state_dict()[
#         'attention.self.query.weight']
#     encoder10_query_bias = layer10.state_dict()['attention.self.query.bias']
#     encoder10_key_weight = layer10.state_dict()['attention.self.key.weight']
#     encoder10_key_bias = layer10.state_dict()['attention.self.key.bias']
#     encoder10_value_weight = layer10.state_dict()[
#         'attention.self.value.weight']
#     encoder10_value_bias = layer10.state_dict()['attention.self.value.bias']
#     encoder10_dense1_weight = layer10.state_dict(
#     )['attention.output.dense.weight']
#     encoder10_dense1_bias = layer10.state_dict()['attention.output.dense.bias']
#     encoder10_lnorm_weight = layer10.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder10_lnorm_bias = layer10.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q10 weight: ", encoder10_query_weight.shape)
#     print("Q10 bias: ", encoder10_query_bias.shape)
#     print("LN10 weight: ", encoder10_lnorm_weight.shape)
#     print("LN10 Bias: ", encoder10_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_query_weight.txt'),
#                transform_w(encoder10_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_query_bias.txt'),
#                encoder10_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_key_weight.txt'),
#                transform_w(encoder10_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_key_bias.txt'),
#                encoder10_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_value_weight.txt'),
#                transform_w(encoder10_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_value_bias.txt'),
#                encoder10_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_dense1_weight.txt'),
#                transform_w(encoder10_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_dense1_bias.txt'),
#                encoder10_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_lnorm_weight.txt'),
#                transform_feature(encoder10_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_lnorm_bias.txt'),
#                encoder10_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer10.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer10.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer10.state_dict()['output.dense.weight']
#     ffn2_b = layer10.state_dict()['output.dense.bias']
#     ffn2_lnw = layer10.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer10.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder10_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 11
#     layer11 = model.encoder.layer[10]
#     encoder11_query_weight = layer11.state_dict()[
#         'attention.self.query.weight']
#     encoder11_query_bias = layer11.state_dict()['attention.self.query.bias']
#     encoder11_key_weight = layer11.state_dict()['attention.self.key.weight']
#     encoder11_key_bias = layer11.state_dict()['attention.self.key.bias']
#     encoder11_value_weight = layer11.state_dict()[
#         'attention.self.value.weight']
#     encoder11_value_bias = layer11.state_dict()['attention.self.value.bias']
#     encoder11_dense1_weight = layer11.state_dict(
#     )['attention.output.dense.weight']
#     encoder11_dense1_bias = layer11.state_dict()['attention.output.dense.bias']
#     encoder11_lnorm_weight = layer11.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder11_lnorm_bias = layer11.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q11 weight: ", encoder11_query_weight.shape)
#     print("Q11 bias: ", encoder11_query_bias.shape)
#     print("LN11 weight: ", encoder11_lnorm_weight.shape)
#     print("LN11 Bias: ", encoder11_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_query_weight.txt'),
#                transform_w(encoder11_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_query_bias.txt'),
#                encoder11_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_key_weight.txt'),
#                transform_w(encoder11_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_key_bias.txt'),
#                encoder11_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_value_weight.txt'),
#                transform_w(encoder11_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_value_bias.txt'),
#                encoder11_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_dense1_weight.txt'),
#                transform_w(encoder11_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_dense1_bias.txt'),
#                encoder11_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_lnorm_weight.txt'),
#                transform_feature(encoder11_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_lnorm_bias.txt'),
#                encoder11_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer11.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer11.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer11.state_dict()['output.dense.weight']
#     ffn2_b = layer11.state_dict()['output.dense.bias']
#     ffn2_lnw = layer11.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer11.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder11_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')

#     # weight+bias save - encoder 12
#     layer12 = model.encoder.layer[11]
#     encoder12_query_weight = layer12.state_dict()[
#         'attention.self.query.weight']
#     encoder12_query_bias = layer12.state_dict()['attention.self.query.bias']
#     encoder12_key_weight = layer12.state_dict()['attention.self.key.weight']
#     encoder12_key_bias = layer12.state_dict()['attention.self.key.bias']
#     encoder12_value_weight = layer12.state_dict()[
#         'attention.self.value.weight']
#     encoder12_value_bias = layer12.state_dict()['attention.self.value.bias']
#     encoder12_dense1_weight = layer12.state_dict(
#     )['attention.output.dense.weight']
#     encoder12_dense1_bias = layer12.state_dict()['attention.output.dense.bias']
#     encoder12_lnorm_weight = layer12.state_dict(
#     )['attention.output.LayerNorm.weight']
#     encoder12_lnorm_bias = layer12.state_dict(
#     )['attention.output.LayerNorm.bias']
#     print("Q12 weight: ", encoder12_query_weight.shape)
#     print("Q12 bias: ", encoder12_query_bias.shape)
#     print("LN12 weight: ", encoder12_lnorm_weight.shape)
#     print("LN12 Bias: ", encoder12_lnorm_bias.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_query_weight.txt'),
#                transform_w(encoder12_query_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_query_bias.txt'),
#                encoder12_query_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_key_weight.txt'),
#                transform_w(encoder12_key_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_key_bias.txt'),
#                encoder12_key_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_value_weight.txt'),
#                transform_w(encoder12_value_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_value_bias.txt'),
#                encoder12_value_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_dense1_weight.txt'),
#                transform_w(encoder12_dense1_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_dense1_bias.txt'),
#                encoder12_dense1_bias, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_lnorm_weight.txt'),
#                transform_feature(encoder12_lnorm_weight), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_lnorm_bias.txt'),
#                encoder12_lnorm_bias, fmt='%f')

#     # intermediate data save
#     ffn1_w = layer12.state_dict()['intermediate.dense.weight']
#     ffn1_b = layer12.state_dict()['intermediate.dense.bias']
#     ffn2_w = layer12.state_dict()['output.dense.weight']
#     ffn2_b = layer12.state_dict()['output.dense.bias']
#     ffn2_lnw = layer12.state_dict()['output.LayerNorm.weight']
#     ffn2_lnb = layer12.state_dict()['output.LayerNorm.bias']
#     print("ffn1 weight out : ", ffn1_w.shape)
#     print("ffn1 bias out : ", ffn1_b.shape)
#     print("ffn2 weight out : ", ffn2_w.shape)
#     print("ffn2 bias out : ", ffn2_b.shape)
#     print("ffn2 ln weight out : ", ffn2_lnw.shape)
#     print("ffn2 ln bias out : ", ffn2_lnb.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn1_weight.txt'),
#                transform_w(ffn1_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn1_bias.txt'),
#                ffn1_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn2_weight.txt'),
#                transform_w(ffn2_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn2_bias.txt'),
#                ffn2_b, fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn2_lnweight.txt'),
#                transform_feature(ffn2_lnw), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'encoder12_ffn2_lnbias.txt'),
#                ffn2_lnb, fmt='%f')
#   # pooler data
#     # hidden_states0 = embedding_output
#     # for layer_module in model.encoder.layer:
#     #   # MHA + LayerNorm
#     #     attention_out = attention(layer_module, hidden_states0)
#     #     ff_out = ffn(layer_module, attention_out)
#     #     hidden_states0 = ff_out
#     # sequence_output = hidden_states0
#     # # np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_in.txt'),
#     # #            transform_feature(sequence_output), fmt='%f')
#     # print(sequence_output.shape)
#     # pooler_layer_out = model.pooler(sequence_output)
#     # print(pooler_layer_out.shape)
#     print("last_hidden_state out : ", output.last_hidden_state.shape)
#     pooler_layer_out = model.pooler(output.last_hidden_state)

#     pooler_in = output.last_hidden_state
#     first_token_tensor = pooler_in[:, 0]
#     first_token_tensor = torch.flatten(first_token_tensor)
#     pooler_in = torch.flatten(pooler_in)
#     print("first_token out : ", first_token_tensor.shape)

#     pooler_out_gt = output.pooler_output
#     pooler_out_gt = torch.flatten(pooler_out_gt)
#     print("pooler out : ", output.pooler_output.shape)
#     print("pooler out gt: ", pooler_out_gt.shape)
#     print("pooler in : ", pooler_in.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_first_token_in.txt'),
#                first_token_tensor.detach().numpy(), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_in.txt'),
#                pooler_in.detach().numpy(), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_out_gt.txt'),
#                pooler_out_gt.detach().numpy(), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_out.txt'),
#                pooler_layer_out.detach().numpy(), fmt='%f')
#     pooler_w = model.state_dict()['pooler.dense.weight']
#     pooler_b = model.state_dict()['pooler.dense.bias']
#     print("pooler weight out : ", pooler_w.shape)
#     print("pooler bias out : ", pooler_b.shape)
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_weight.txt'),
#                transform_feature(pooler_w), fmt='%f')
#     np.savetxt(os.path.join(txt_dir, txt_dir + 'pooler_bias.txt'),
#                pooler_b, fmt='%f')
