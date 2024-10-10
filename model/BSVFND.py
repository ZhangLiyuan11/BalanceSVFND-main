from model.Mambaformer import *
from utils.tools import *

class FusionNet(nn.Module):
    def __init__(self,trans_dim,fea_dim):
        super(FusionNet, self).__init__()
        self.fea_dim = fea_dim
        self.trans_dim = trans_dim
        self.linear = nn.Linear(self.trans_dim, self.fea_dim) 

    def forward(self, text_fea, video_fea, audio_fea):
        fused_fea = torch.mean(torch.stack((text_fea,video_fea,audio_fea),dim=0), dim=0)
        # linear and relu
        fea = F.relu(self.linear(fused_fea))
        return fea
    
class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.adaptive_max_pool_layer = nn.AdaptiveMaxPool1d(output_size=1)
        
    def forward(self, fea):
        pooled_sequence = self.adaptive_max_pool_layer(fea.transpose(1, 2)).transpose(1, 2).squeeze(dim=-2)
        return pooled_sequence


class BSVFNDModel(torch.nn.Module):
    def __init__(self,fea_dim,dropout,dataset):
        super(BSVFNDModel, self).__init__()
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768

        self.img_dim = 1024
        self.hubert_dim = 1024
        self.dim = fea_dim
        self.num_heads = 8
        self.trans_dim = 512
        self.dropout = dropout

        # self.maxpool = MaxPool()
        
        # self.co_attention_a = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        # self.co_attention_v = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        # self.co_attention_t = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)

        self.CrossMambaformer = CrossMambaformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                                          dropout_probability=self.dropout)
        # self.CrossMambaformer = Attention_Mamba(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
        #                                   dropout_probability=self.dropout)
        
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_hubert = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        self.fusion_net = FusionNet(self.trans_dim,self.dim)
        self.classifier = nn.Linear(fea_dim,2)

    def forward(self, **kwargs):
        ### Title ###
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask=kwargs['title_mask']#(batch,512)
        fea_text=self.bert(title_inputid,attention_mask=title_mask)['last_hidden_state']#(batch,sequence,768)
        fea_text = self.linear_text(fea_text)
        
        ### Audio Frames ###
        audio_feas = kwargs['audio_feas']
        fea_audio = self.linear_hubert(audio_feas)

        ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        fea_img = self.linear_img(frames)

        # fea_text = self.co_attention_t(fea_text,fea_img)
        # fea_img = self.co_attention_v(fea_img,fea_text)
        # fea_audio = self.co_attention_a(fea_audio,fea_text)
        
        fea_text,fea_img,fea_audio = self.CrossMambaformer(fea_text,fea_img,fea_audio)

        fea_audio = torch.mean(fea_audio, -2)
        fea_text = torch.mean(fea_text, -2)
        fea_img = torch.mean(fea_img, -2)
        
        final_fea = self.fusion_net(fea_text,fea_img,fea_audio)

        output = self.classifier(final_fea)

        return output
