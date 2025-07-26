import torch
import torch.nn as nn
import torch.optim as optim
from nltk import accuracy
from statsmodels.tsa.ardl.pss_critical_values import crit_vals
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from configs import Config
from models.TextCNN import TextCNN
from dataset import TextDataset
from utils.vocabulary import Vocabulary
from utils.preprocess import load_data_and_split, build_glove_embedding_matrix, load_test_data

def get_optimizer(model, config):
    if config.optimizer=="Adam":
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_deacy)
    elif config.optimizer=="SGD":
        return optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError("Invalid optimizer")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss=0
    total_correct=0
    total_samples=0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels=texts.to(device), labels.to(device)
            outputs=model(texts)
            loss=criterion(outputs, labels)
            total_loss+=loss.item()
            _, predicted=torch.max(outputs.data, 1)
            total_samples+=labels.size(0)
            total_correct+=(predicted==labels).sum().item()
    avg_loss=total_loss/len(dataloader)
    accuracy=100*total_correct/total_samples
    return avg_loss, accuracy

def run_training(config:Config):
    start_time=time.time()
    print("Loading and preprocessing data")
    train_texts, val_texts, train_labels, val_labels=load_data_and_split(config.train_path)

    if os.path.exists(config.vocab_path):
        vocab=Vocabulary.load(config.vocab_path)
    else:
        all_train_texts=train_texts+val_texts#使用全部训练数据构建词表
        vocab=Vocabulary.build_from_sentences(all_train_texts, min_freq=config.min_word_freq)
        vocab.save(config.vocab_path)
    config.vocab_size=len(vocab)
    print(f"Vocabulary size:{config.vocab_size}")

    #创建Dataset和DataLoader
    train_dataset=TextDataset(train_texts, train_labels, vocab, config.max_seq_length)
    val_dataset=TextDataset(val_texts, val_labels, vocab, config.max_seq_length)
    train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    #加载Glove Embedding
    embedding_matrix=None
    if config.use_glove:
        embedding_matrix=build_glove_embedding_matrix(config.glvoe_path, vocab, config.embedding_dim)
        if embedding_matrix is None:
            return

    #初始化模型
    model=TextCNN(config, embedding_matrix).to(config.device)
    criterion=nn.CrossEntropyLoss()
    optimizer=get_optimizer(model, config)

    print(f"\n --------Training Details-----------")
    print(f"Model:{config.model_name}")
    print(f"Optimizer:{config.optimizer}")
    print(f"Use Glove:{config.use_glove}, Freeze embeddings:{config.freeze_embeddings}")
    print("-------------------\n")

    #训练循环
    best_val_acc=0.0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss=0
        progress_bar=tqdm(train_loader, desc=f"Epoch{epoch+1}/{config.num_epochs}")
        for texts, labels in progress_bar:
            texts, labels=texts.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs=model(texts)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
            progress_bar.set_postfix({'loss':loss.item()})

        avg_train_loss=epoch_loss/len(train_loader)

        #评估
        val_loss, val_acc=evaluate(model, val_loader, criterion, config.device)
        print(f"Epoch:{epoch+1}/{config.num_epochs}, Train Loss:{avg_train_loss:.4f}, Val Loss:{val_loss:.4f}, Val Acc:{val_acc:.2f}%")

        #保存最佳模型
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            if not os.path.exists(config.model_save_dir):
                os.makedirs(config.model_save_dir)

            model_filename=f"best_model_lr{config.learning_rate}_opt{config.optimizer}.pth"
            save_path=os.path.join(config.model_save_dir, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with accuracy: {best_val_acc:.2f}%")

    end_time=time.time()
    training_duration=end_time-start_time
    print(f"\nTraining finished in {training_duration:.2f} seconds.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc, training_duration

if __name__ =="__main__":
    config=Config()
    run_training(config)