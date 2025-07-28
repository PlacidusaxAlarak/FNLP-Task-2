import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)

        position=torch.arange(max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)*(-math.log(10000.0)/d_model))
        pe=torch.zeros(max_len, d_model)
        pe[:, 0::2]=torch.sin(position*div_term)
        pe[:, 1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x=x+self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextTransformer(nn.Module):
    def __init__(self, config, embedding_matrix=None):
        super(TextTransformer, self).__init__()
        self.config=config
        self.embedding_dim=config.embedding_dim

        #Embedding Layer
        self.embedding=nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        if config.use_glove and embedding_matrix is not None:
            print("Initializing Transformer embedding layer with Glove weights")
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad=not config.freeze_embeddings
        else:
            print("Initializing Transformer embedding layer with random weights")
            self.embedding.weight.requires_grad=True

        #Positional Encoding
        self.pos_encoder=PositionalEncoding(config.embedding_dim, config.dropout_prob, config.max_seq_length)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,#输入输出维度
            nhead=config.trans_nhead,#必须能被d_model整除
            dim_feedforward=config.trans_dim_feedforward,#中间前馈层的维度
            dropout=config.dropout_prob,
            batch_first=True
        )

        self.transformer_encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.trans_num_layers
        )

        self.fc=nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x):
        padding_mask=(x==0)
        #padding_mask:[batch_size, seq_len], True的地方是填充位置
        embedded=self.embedding(x)*math.sqrt(self.embedding_dim)
        pos_encoded=self.pos_encoder(embedded)

        encoded=self.transformer_encoder(pos_encoded, src_key_padding_mask=padding_mask)

        encoded[padding_mask]=0#填充位置的输出置为0
        pooled=encoded.sum(dim=1)/(~padding_mask).sum(dim=1, keepdim=True).clamp(min=1.0)#clamp任何小于1.0的会被提升到1.0
        return self.fc(pooled)