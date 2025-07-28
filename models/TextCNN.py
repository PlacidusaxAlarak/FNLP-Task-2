import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config, embedding_matrix=None):
        super(TextCNN, self).__init__()
        self.config=config

        #Embedding Layer
        self.embedding=nn.Embedding(config.vocab_size, config.embedding_dim)
        if config.use_glove and embedding_matrix is not None:
            print("Initailizing embedding layer with glove weights")
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad=not config.freeze_embeddings
        else:
            print("正在使用随机权重初始化Embedding层")
            self.embedding.weight.requires_grad=True#模型可以学习这个参数
        #Convolutional Layers
        self.convs=nn.ModuleList([
            nn.Conv1d(in_channels=config.embedding_dim, out_channels=config.num_filters, kernel_size=k)
            for k in config.filter_sizes
        ])

        #Fully Connectec Layer
        self.fc=nn.Linear(config.num_filters*len(config.filter_sizes), config.num_classes)

        #Dropout
        self.dropout=nn.Dropout(config.dropout_prob)
    def forward(self, x):
        #x:[batch_size, seq_len]
        embedded=self.embedding(x)#->[batch_size, seq_len, embedding_dim]

        embedded=embedded.permute(0, 2, 1)#交换两个维度，变成[batch_size, embedding_dim, seq_len], 第二个维度是输入通道数

        conved=[F.relu(conv(embedded)) for conv in self.convs]

        pooled=[F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]#去掉大小为1的维度

        cat=torch.cat(pooled, dim=1)

        dropped=self.dropout(cat)

        output=self.fc(dropped)

        return output

