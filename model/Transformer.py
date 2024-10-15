import math
import copy
import torch
import torch.nn as nn

from Mambaformer import PoswiseFeedForwardNet,FeedForward

class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        # All of these will get deep-copied multiple times internally
        # mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability,number_of_heads)
        self.encoder = Encoder(encoder_layer)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters 
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input1, input2):
        # Q:input1 K,V:input2
        src_representations_batch1 = self.encoder(input1, input2)
        return src_representations_batch1

class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layer = encoder_layer
        self.ffn_layer = PoswiseFeedForwardNet(encoder_layer.model_dimension,encoder_layer.model_dimension*2)
        # self.ffn_layer = FeedForward(encoder_layer.model_dimension,encoder_layer.model_dimension*2)
    def forward(self, src1, src2):
        # Forward pass through the encoder stack
        src_representations_batch = self.encoder_layer(src1, src2)
        representations = self.ffn_layer(src_representations_batch)
        return representations

class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability,number_of_heads):
        super().__init__()
        self.sublayer1 = SublayerLogic(model_dimension, dropout_probability)
        # self.sublayer2 = SublayerLogic(model_dimension, dropout_probability)
        self.mha1 = MultiHeadedAttention(model_dimension=model_dimension,number_of_heads=number_of_heads,dropout_probability=dropout_probability,log_attention_weights=False)
        # self.mha2 = MultiHeadedAttention(model_dimension=model_dimension,number_of_heads=number_of_heads,dropout_probability=dropout_probability,log_attention_weights=False)
        self.model_dimension = model_dimension
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, srb1, srb2):
        encoder_self_attention1 = lambda srb1, srb2: self.mha1(query=srb1, key=srb2, value=srb2)
        # encoder_self_attention2 = lambda srb1, srb2: self.mha2(query=srb1, key=srb2, value=srb2)
        src_representations_batch = self.norm(self.sublayer1(srb1, srb2, encoder_self_attention1))
        # src_representations_batch = self.norm(self.sublayer2(src_representations_batch, srb2, encoder_self_attention2))
        return src_representations_batch

class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
        # self.pos_emb_v = PosEncoding(input1_len * 10, model_dimension)
        # self.pos_emb_s = PosEncoding(input2_len * 10, model_dimension)
    def forward(self, srb1, srb2, mha):
        return srb1 + self.dropout(mha(self.norm(srb1), self.norm(srb2)))

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        token_representations = self.out_projection_net(reshaped)
        return token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

