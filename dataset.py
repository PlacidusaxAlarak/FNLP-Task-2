import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts=texts
        self.labels=labels
        self.vocab=vocab
        self.max_len=max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text=self.texts[idx]
        label=self.labels[idx]

        #文本转ID
        unk_idx=self.vocab.word2idx[self.vocab.unk_token]
        text_ids=[self.vocab.word2idx.get(word, unk_idx) for word in text]

        #填充/截断
        pad_idx=self.vocab.word2idx[self.vocab.pad_token]
        if len(text_ids) < self.max_len:
            text_ids.extend([pad_idx] * (self.max_len - len(text_ids)))
        else:
            text_ids=text_ids[:self.max_len]
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)