from model.Mambaformer import *
from utils.tools import *

class FusionNet(nn.Module):
    def __init__(self,fea_dim,dropout):
        super(FusionNet, self).__init__()
        self.fea_dim = fea_dim
        self.dropout = dropout
        self.attention_linear = torch.nn.Linear(self.fea_dim, 1)
        self.classifier = nn.Linear(fea_dim,2)
        
    def forward(self, fused_fea, stage):
        if stage == 1:
            final_fea = torch.mean(fused_fea,dim=0)
            output = self.classifier(final_fea)
        else:
            att_weights = F.softmax(self.attention_linear(fused_fea.permute((1, 0, 2))),dim=1) + 1
            feas = att_weights * fused_fea.permute((1, 0, 2))
            final_fea = (feas[:, 0, :] + feas[:, 1, :] + feas[:, 2, :]) / 4
            output = self.classifier(final_fea)
        return output 
    
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
        self.linear_hubert = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        
        self.trans_linear = nn.Sequential(torch.nn.Linear(self.trans_dim, self.dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # self.trans_linear = torch.nn.Linear(self.trans_dim, self.dim)
        self.fusion_net = FusionNet(self.dim,self.dropout)
        
    def forward(self, stage, **kwargs):
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
        
        fused_fea = torch.stack((fea_text,fea_img,fea_audio),dim=0)
        fused_fea = self.trans_linear(fused_fea)
        
        output = self.fusion_net(fused_fea,stage)

        return output
