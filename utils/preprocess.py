# preprocess.py (最终正确版 - 适用于无表头数据集)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def load_data_and_split(file_path, test_size=0.1):
    """加载数据并划分为训练集和验证集"""
    try:
        # 🔥 关键修改：明确指定文件没有表头，并直接为列命名
        df = pd.read_csv(file_path, sep='\t', header=None, names=['sentence', 'label'])
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        print("请确保文件是有效的、用制表符分隔的两列文件。")
        return None, None, None, None

    # 确保标签是整数类型
    df['label'] = df['label'].astype(int)
    sentences = df['sentence'].fillna('').apply(lambda x: str(x).lower().split()).tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=test_size, random_state=42, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels


def load_test_data(filepath):
    """加载测试数据 (逻辑与上面保持一致)"""
    try:
        # 🔥 关键修改：同样地，没有表头，直接命名
        df = pd.read_csv(filepath, sep='\t', header=None, names=['sentence', 'label'])
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        return None, None

    df['label'] = df['label'].astype(int)
    sentences = df['sentence'].fillna('').apply(lambda x: str(x).lower().split()).tolist()
    labels = df['label'].tolist()
    return sentences, labels


def build_glove_embedding_matrix(glove_path, vocab, embedding_dim):
    """构建GloVe词向量矩阵"""
    print("正在加载GloVe词向量...")
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取GloVe文件"):
            values = line.split()
            word = values[0]
            if len(values) == embedding_dim + 1:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    found_words = 0
    for word, i in vocab.word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    print(f"在GloVe词汇表中找到了 {found_words}/{len(vocab)} 个单词。")
    return embedding_matrix