import torch
import torch.nn as nn
import math


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

        #一个可学习的CLS token嵌入
        self.cls_token=nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        #Positional Encoding
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.embedding_dim)
        self.layer_norm=nn.LayerNorm(config.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,#输入输出维度
            nhead=config.trans_nhead,#必须能被d_model整除
            dim_feedforward=config.trans_dim_feedforward,#中间前馈层的维度
            dropout=config.dropout_prob,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.trans_num_layers
        )
        #分类器头
        self.classifier=nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim//2),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.embedding_dim//2, config.num_classes)
        )


    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size = x.size(0)
        # 注意: config中的max_seq_length是最终送入encoder的长度
        # 我们需要从输入x中截取 max_seq_length - 1 个词元
        seq_len_words = self.config.max_seq_length - 1
        
        # 1. 截取输入序列，为[CLS] token腾出空间
        words_tensor = x[:, :seq_len_words]

        # 2. 为截取后的词元创建正确的padding mask
        # (words_tensor == 0) 的地方为True, 表示是padding，需要被mask
        word_padding_mask = (words_tensor == 0)
        
        # 3. 为[CLS] token创建mask (全False, 因为CLS token从不被mask)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        
        # 4. 拼接成最终的padding mask
        # 最终mask的长度是 1 (for CLS) + (max_seq_length - 1) = max_seq_length
        padding_mask = torch.cat([cls_mask, word_padding_mask], dim=1)

        # 5. 获取词嵌入
        word_embeddings = self.embedding(words_tensor)

        # 6. 准备[CLS] token的嵌入，并扩展到整个batch
        cls_token_embeddings = self.cls_token.expand(batch_size, -1, -1)

        # 7. 将[CLS]嵌入拼接到词嵌入序列的开头
        embeddings = torch.cat([cls_token_embeddings, word_embeddings], dim=1)
        
        # 8. 添加位置嵌入
        # 最终序列长度是 config.max_seq_length
        position_ids = torch.arange(self.config.max_seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 9. 将带有[CLS]的序列送入Transformer Encoder
        encoded_layers = self.transformer_encoder(
            src=embeddings,
            src_key_padding_mask=padding_mask
        )

        # 10. 提取[CLS] token的输出作为整个序列的表示
        cls_token_output = encoded_layers[:, 0, :]
        logits = self.classifier(cls_token_output)

        return logits