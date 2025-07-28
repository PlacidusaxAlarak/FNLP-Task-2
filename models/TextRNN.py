import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, config, embedding_matrix=None):
        super(TextRNN, self).__init__()
        self.config=config

        #Embedding Layer
        self.embedding=nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)#索引为0的输入是填充符
        if config.use_glove and embedding_matrix is not None:
            print("Initializing RNN embedding layer with Glove weights")
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad=not config.freeze_embeddings
        else:
            print("Initializing RNN embedding layer with random weights")
            self.embedding.weight.requires_grad=True

        #LSTM Layer
        self.lstm=nn.LSTM(
            input_size=config.embedding_dim,#输入维度必须等于词向量的维度
            hidden_size=config.rnn_hidden_size,#隐藏状态维度
            num_layers=config.rnn_num_layers,
            bidirectional=config.rnn_bidirectional,#是否使用双向LSTM
            batch_first=True,#[BATCH_SIZE, SEQ_LEN, FEATURES]
            dropout=config.dropout_prob if config.rnn_num_layers>1 else 0#多层LSTM之间添加dropout
        )

        #Connected Layer
        fc_input_size=config.rnn_hidden_size*(2 if config.rnn_bidirectional else 1)#如果双向那么就需要拼起来
        self.fc=nn.Linear(fc_input_size, config.num_classes)

        #Dropout
        self.dropout=nn.Dropout(config.dropout_prob)

    def forward(self, x):
        #x:[batch_size, seq_len]
        embedded=self.embedding(x)
        #embedded:[batch_size, seq_len, embedding_dim]
        lstm_out, (h_n, c_n)=self.lstm(embedded)
        #lstm_out:[batch_size, seq_len, hidden_size*num_directions(2 or 1)]
        #h_n:只包含最后一个时间步的隐藏状态[num_layers*num_directions. batch_size, heddien_size]
        #c_n:最后一个时间步的隐藏状态
        if self.config.rnn_bidirectional:
            hidden=torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)#前向和后向的隐藏状态拼起来
        else:
            hidden=h_n[-1, :, :]

        dropped=self.dropout(hidden)
        output=self.fc(dropped)
        return output