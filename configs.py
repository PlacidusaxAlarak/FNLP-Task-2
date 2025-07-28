import torch

class Config:
    def __init__(self):
        #数据集和路径
        self.train_path="./data/new_train.tsv"
        self.test_path="./data/new_test.tsv"
        self.vocab_path="./data/vocab.pkl"#转化成唯一的数字
        self.model_save_dir="./saved_models"
        self.result_csv_path="./results/experiment_results.csv"

        #GPU设置
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #CNN模型超参数
        self.model_name="TextCNN"
        self.embedding_dim=100#与glove一致
        self.num_filters=128#卷积核数量
        self.dropout_prob=0.7
        self.filter_sizes=[3, 4, 5]#卷积核尺寸
        self.weight_deacy=1e-5

        #RNN(LSTM)模型超参数
        self.rnn_hidden_size=128
        self.rnn_num_layers=2
        self.rnn_bidirectional=True

        #Transformer超参数
        self.trans_nhead=4
        self.trans_num_layers=2
        self.trans_dim_feedforward=512
        #训练超参数
        self.num_epochs=50
        self.batch_size=8
        self.learning_rate=1e-4
        self.optimizer='Adam'
        self.loss_function="CrossEntropyLoss"

        #glove预训练词向量
        self.use_glove=True
        self.glvoe_path="data/glove.6B.100d.txt"
        self.freeze_embeddings=False#是否冻结embeddings层

        #数据预处理
        self.max_seq_length=128#句子最大长度
        self.min_word_freq=1#构建词表的最小频率
        self.num_classes=5#类别数（5种情绪）
        self.vocab_size=0#构建词表后更新

    def update_params(self, **kwargs):  # 表示函数的参数可以是任意数量的关键字参数，传入时会以字典的形式保存bi
        # 用于脚本动态修改参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if 'embedding_dim' in kwargs:
            self.glove_path = f"data/glove.6B.{kwargs['embedding_dim']}d.txt"
