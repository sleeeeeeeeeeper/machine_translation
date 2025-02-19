import pandas as pd
from collections import Counter
import pickle
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def build_vocab(df, col_name, vocab_size, save_path=None):
    tokens = [token for line in df[col_name] for token in eval(line)]
    token_counts = Counter(tokens)
    most_common_tokens = token_counts.most_common(vocab_size - 4)
    vocab = {token: i for i, (token, _) in enumerate(most_common_tokens, start=4)}
    vocab['<sos>'] = 0 # start of sentence
    vocab['<eos>'] = 1 # end of sentence
    vocab['<pad>'] = 2 # 填充词
    vocab['<unk>'] = 3 # 未知词
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"词汇表已保存到 {save_path}")
    print(vocab)
    return vocab

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# 构建数据集，将英文和中文句子转换为索引
def build_data(df, english_vocab, chinese_vocab):
    # 重置索引，使索引从0开始
    df = df.reset_index(drop=True)
    df = df.sample(n=100000).reset_index(drop=True)
    english = [[0] + [english_vocab.get(token, 3) for token in eval(line)] + [1] for line in df['english']] # .get()方法返回指定键的值，如果值不在字典中返回默认值
    chinese = [[0] + [chinese_vocab.get(token, 3) for token in eval(line)] + [1] for line in df['chinese']] # 因此，未知词的索引赋为了3
    print(df.iloc[0])
    print(english[0])
    print(chinese[0])
    return english, chinese

# TED数据集
class TED:
    def __init__(self, file_path, english_vocab=None, chinese_vocab=None, train=True):
        self.name = 'TED2020'
        self.df = pd.read_csv(file_path)
        self.english_vocab = build_vocab(self.df, 'english', 8000, './data/english_vocab.pkl') if english_vocab is None else english_vocab
        self.chinese_vocab = build_vocab(self.df, 'chinese', 8000, './data/chinese_vocab.pkl') if chinese_vocab is None else chinese_vocab
        self.id2english = {i: token for token, i in self.english_vocab.items()}
        self.id2chinese = {i: token for token, i in self.chinese_vocab.items()}
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.1, random_state=42)
        if train:
            self.data = build_data(self.train_df, self.english_vocab, self.chinese_vocab)
        else:
            self.data = build_data(self.test_df, self.english_vocab, self.chinese_vocab)
        self.en_max_len = max(self.df['eng_length'])
        print(f'en_max_len:{self.en_max_len}')
        self.cn_max_len = max(self.df['ch_length'])
        print(f'cn_max_len:{self.cn_max_len}')

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        X, Y = self.data
        X[idx] = X[idx] + [2] * (self.en_max_len - len(X[idx]))
        Y[idx] = Y[idx] + [2] * (self.cn_max_len - len(Y[idx]))
        return torch.tensor(X[idx]), torch.tensor(Y[idx])

# 作为DataLoader的参数，用于对每个batch的数据进行填充
def collate_fn(batch):
    # 对输入的每个样本进行填充，使得它们的长度一致
    X_batch, Y_batch = zip(*batch)  # 拆分输入和目标
    X_batch = pad_sequence(X_batch, batch_first=True, padding_value=2)  # 填充输入
    Y_batch = pad_sequence(Y_batch, batch_first=True, padding_value=2)  # 填充目标
    return X_batch, Y_batch

def load_data(data_name, train=True):
    if data_name == 'ted':
        if train:
            data = TED('./data/preprocessed_data.csv', train=True)
        else:
            data = TED('./data/preprocessed_data.csv', train=False)
    else:
        raise NotImplementedError(f'Unknown data name:{data_name}')
    print(data.name)
    print(f'data:{data.__len__()}')
    return data

if __name__ == '__main__':
    # 测试数据集
    ted = TED('./data/preprocessed_data.csv')
    print(ted.__getitem__(0)) # 测试__getitem__方法
    print(ted.__len__()) # 测试__len__方法
