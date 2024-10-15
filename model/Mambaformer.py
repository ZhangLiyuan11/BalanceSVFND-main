import math
import copy
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class FeedForward(nn.Module):
    def __init__(self, model_dimension, d_ff, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Linear(model_dimension, d_ff)
        self.ff2 = nn.Linear(d_ff, model_dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, representations_batch):
        return self.norm(representations_batch + self.ff2(self.dropout(F.relu(self.ff1(representations_batch)))))

class Mamba_Layer(nn.Module):
    def __init__(self, mamba, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)
        return x

class CrossMambaformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability,d_state=16,d_conv=4):
        super().__init__()
        self.Mamba_text1 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_audio1 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_video1 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_text2 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_audio2 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_video2 = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        
        self.text_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.text_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.video_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.video_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.audio_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.audio_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        
        self.text_sublayer_1 = SublayerLogic(model_dimension, dropout_probability)
        self.text_self_mha1 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.video_sublayer_1 = SublayerLogic(model_dimension, dropout_probability)
        self.video_self_mha1 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.audio_sublayer_1 = SublayerLogic(model_dimension, dropout_probability)
        self.audio_self_mha1 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)

        self.text_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.text_self_mha2 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.video_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.video_self_mha2 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.audio_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.audio_self_mha2 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)

        self.text_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        self.video_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        self.audio_ffn_layer = PoswiseFeedForwardNet(model_dimension, model_dimension * 2)
        self.init_params()
        self.norm = nn.LayerNorm(model_dimension)

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters 
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text_input, video_input,audio_input):
        # Mamba
        text_input = self.Mamba_text1(text_input)
        video_input = self.Mamba_video1(video_input)
        audio_input = self.Mamba_audio1(audio_input)
        # self_attention
        text_self_attention1 = lambda srb1, srb2: self.text_self_mha1(query=srb1, key=srb2, value=srb2)
        video_self_attention1= lambda srb1, srb2: self.video_self_mha1(query=srb1, key=srb2, value=srb2)
        audio_self_attention1 = lambda srb1, srb2: self.audio_self_mha1(query=srb1, key=srb2, value=srb2)
        text_representations_batch = self.norm(self.text_sublayer_1(text_input, text_input, text_self_attention1))
        video_representations_batch = self.norm(self.video_sublayer_1(video_input, video_input, video_self_attention1))
        audio_representations_batch = self.norm(self.audio_sublayer_1(audio_input, audio_input, audio_self_attention1))
        # cross_attention
        text_cross_attention = lambda srb1, srb2: self.text_mha(query=srb1, key=srb2, value=srb2)
        video_cross_attention = lambda srb1, srb2: self.video_mha(query=srb1, key=srb2, value=srb2)
        audio_cross_attention = lambda srb1, srb2: self.audio_mha(query=srb1, key=srb2, value=srb2)
        text_representations_batch = self.norm(self.text_sublayer(text_representations_batch, video_representations_batch, text_cross_attention))
        video_representations_batch = self.norm(self.video_sublayer(video_representations_batch, text_representations_batch, video_cross_attention))
        audio_representations_batch = self.norm(self.audio_sublayer(audio_representations_batch, text_representations_batch, audio_cross_attention))
        # Mamba
        text_representations_batch = self.Mamba_text2(text_representations_batch)
        video_representations_batch = self.Mamba_video2(video_representations_batch)
        audio_representations_batch = self.Mamba_audio2(audio_representations_batch)
        # self_attention
        text_self_attention2 = lambda srb1, srb2: self.text_self_mha2(query=srb1, key=srb2, value=srb2)
        video_self_attention2 = lambda srb1, srb2: self.video_self_mha2(query=srb1, key=srb2, value=srb2)
        audio_self_attention2 = lambda srb1, srb2: self.audio_self_mha2(query=srb1, key=srb2, value=srb2)
        text_representations_batch = self.norm(self.text_sublayer_2(text_representations_batch, text_representations_batch, text_self_attention2))
        video_representations_batch = self.norm(self.video_sublayer_2(video_representations_batch, video_representations_batch, video_self_attention2))
        audio_representations_batch = self.norm(self.audio_sublayer_2(audio_representations_batch, audio_representations_batch, audio_self_attention2))
        # ffn
        text_representations_batch = self.text_ffn_layer(text_representations_batch)
        video_representations_batch = self.video_ffn_layer(video_representations_batch)
        audio_representations_batch = self.audio_ffn_layer(audio_representations_batch)
        return text_representations_batch,video_representations_batch,audio_representations_batch

    
class Attention_Mamba(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability,d_state=16,d_conv=4):
        super().__init__()
        self.Mamba_text = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_audio = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)
        self.Mamba_video = Mamba_Layer(Mamba(model_dimension, d_state=d_state, d_conv=d_conv), model_dimension)

        self.text_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.text_self_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.video_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.video_self_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.audio_sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.audio_self_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)

        self.text_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.text_cross_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.video_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.video_cross_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)
        self.audio_sublayer_2 = SublayerLogic(model_dimension, dropout_probability)
        self.audio_cross_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability, log_attention_weights=False)

        self.init_params()
        self.norm = nn.LayerNorm(model_dimension)

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters 
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text_input, video_input,audio_input):
        # self-attention
        text_self_attention = lambda srb1, srb2: self.text_self_mha(query=srb1, key=srb2, value=srb2)
        video_self_attention = lambda srb1, srb2: self.video_self_mha(query=srb1, key=srb2, value=srb2)
        audio_self_attention = lambda srb1, srb2: self.audio_self_mha(query=srb1, key=srb2, value=srb2)
        text_representations_batch = self.norm(self.text_sublayer(text_input, text_input, text_self_attention))
        video_representations_batch = self.norm(self.video_sublayer(video_input, video_input, video_self_attention))
        audio_representations_batch = self.norm(self.audio_sublayer(audio_input, audio_input, audio_self_attention))
        # cross-attention
        text_cross_attention = lambda srb1, srb2: self.text_cross_mha(query=srb1, key=srb2, value=srb2)
        video_cross_attention = lambda srb1, srb2: self.video_cross_mha(query=srb1, key=srb2, value=srb2)
        audio_cross_attention = lambda srb1, srb2: self.audio_cross_mha(query=srb1, key=srb2, value=srb2)
        text_representations_batch = self.norm(self.text_sublayer_2(text_representations_batch, video_representations_batch, text_cross_attention))
        video_representations_batch = self.norm(self.video_sublayer_2(video_representations_batch, text_representations_batch, video_cross_attention))
        audio_representations_batch = self.norm(self.audio_sublayer_2(audio_representations_batch, text_representations_batch, audio_cross_attention))
        # Mamba
        text_representations_batch = self.Mamba_text(text_representations_batch)
        video_representations_batch = self.Mamba_video(video_representations_batch)
        audio_representations_batch = self.Mamba_audio(audio_representations_batch)
        return text_representations_batch,video_representations_batch,audio_representations_batch
    
    
class CrossSublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
    def forward(self, srb1, srb2, mha):
        return srb1 * 2 + self.dropout(mha(self.norm(srb1), self.norm(srb2)))
    
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
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
        # query b*300*1024, key b*49*1024, value b*49*1024
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        # query b*8*300*128, key b*8*49*128, value b*8*49*128
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
