import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from dataset import load_vocab
import fasttext.util


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, pretrained_embeddings=None):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # 默认的嵌入层
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)  # 使用预训练词向量初始化
            self.embedding.weight.requires_grad = False  # 冻结嵌入层

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=0.2)  # LSTM层

    def forward(self, X):
        # print('Encoder forward')
        embedded = self.embedding(X)
        # print(f'embedded shape: {embedded.shape}') # 形状为(seq_len, batch_size, emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded)  # LSTM的输出: 输出序列, (隐藏状态, 细胞状态)
        # print(f'outputs shape: {outputs.shape}') # 输出序列的形状为(seq_len, batch_size, num_directions * hidden_size)
        # print(f'hidden shape: {hidden.shape}') # 隐藏状态的形状为(num_layers * num_directions, batch_size, hidden_size)
        # print(f'cell shape: {cell.shape}') # 细胞状态的形状为(num_layers * num_directions, batch_size, hidden_size)
        return hidden, cell  # 返回隐藏状态和细胞状态

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, pretrained_embeddings=None):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)  # 使用预训练词向量初始化
            self.embedding.weight.requires_grad = False  # 冻结嵌入层
        
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=0.2)  # 解码器LSTM
        self.fc_out = nn.Linear(hid_dim, output_dim)  # 输出层，将LSTM的输出转换为词汇空间

    def forward(self, input, hidden, cell):
        # print('Decoder forward')
        embedded = self.embedding(input)
        # print(f'embedded shape: {embedded.shape}')
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # 通过LSTM计算下一个输出
        # print(f'output shape: {output.shape}')
        # print(f'hidden shape: {hidden.shape}')
        # print(f'cell shape: {cell.shape}')
        prediction = self.fc_out(output)  # 通过全连接层输出预测结果
        return prediction, hidden, cell  # 返回预测结果、隐藏状态和细胞状态

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, zh_embeddings=None, en_embeddings=None, max_len=50):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, pretrained_embeddings=zh_embeddings)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, pretrained_embeddings=en_embeddings)
        self.max_len = max_len
        self.teacher_forcing_ratio = 0.5

    def forward(self, input, target, teacher_forcing=False):
        batch_size = input.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_dim

        # 通过编码器处理源序列
        encode_hidden, encode_cell = self.encoder(input)

        # 初始化解码器的输入
        decode_hidden, decode_cell = encode_hidden, encode_cell  # 将编码器的隐藏状态和细胞状态作为解码器的初始状态
        decode_input = torch.unsqueeze(target[:, 0], dim=1)  # 取目标序列的第一个token作为初始输入
        # print(f'input shape: {decode_input.shape}')

        # 初始化输出
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        outputs_idx = torch.zeros(batch_size, target_len)

        # 解码过程
        for t in range(1, target_len):
            # 将当前输入token和隐藏状态、细胞状态传入解码器
            decode_output, decode_hidden, decode_cell = self.decoder(decode_input, decode_hidden, decode_cell)
            top_prob, top_idx = decode_output.topk(1)  # 取概率最大的token作为输出
            # 如果使用teacher forcing，生成随机数来决定是否使用真实的目标token作为下一个输入
            if teacher_forcing and torch.rand(1).item() < self.teacher_forcing_ratio:
                decode_input = torch.unsqueeze(target[:, t], dim=1)
            else:
                decode_input = top_idx.squeeze(1)

            output = decode_output.squeeze()
            outputs[:, t, :] = output
            outputs_idx[:, t] = top_idx.squeeze()

        return outputs_idx, outputs


    def translate(self, input, device, temperature=1.0, max_len=50):
        batch_size = input.size(0)
        target_len = max_len
        target_vocab_size = self.decoder.output_dim

        encode_hidden, encode_cell = self.encoder(input)
        decode_hidden, decode_cell = encode_hidden, encode_cell # 使用编码器的隐藏状态和细胞状态作为解码器的初始状态
        decode_input = torch.tensor([0] * batch_size).unsqueeze(1).to(device) # 使用开始标志SOS作为初始输入, 0表示SOS
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        outputs_idx = torch.zeros(batch_size, target_len)
        # 记录每个序列是否结束
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # 解码
        for t in range(1, target_len):
            decode_output, decode_hidden, decode_cell = self.decoder(decode_input, decode_hidden, decode_cell)

            # # 使用温度调节概率分布
            if temperature != 1.0 and temperature != 0.0:
                decode_output = decode_output / temperature  # 调整温度

            probs = F.softmax(decode_output, dim=-1)
            probs = probs.squeeze(1)

            if temperature == 0.0:
                _, top_idx = decode_output.topk(1)
                decode_input = top_idx.squeeze(1)
            else:
                # 否则从概率分布中采样
                top_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                decode_input = top_idx.unsqueeze(1)

            # 更新输出
            output = decode_output.squeeze()
            outputs[:, t, :] = output
            outputs_idx[:, t] = top_idx.squeeze()

            # 如果生成了EOS标记，则结束解码
            eos_mask = (top_idx.squeeze() == 1)  # EOS标记的ID是1
            finished = finished | eos_mask  # 更新已经结束的序列

            if finished.all():
                break

        return outputs_idx


def load_fasttext_embeddings(vocab, model, emb_dim):
    embedding_matrix = np.zeros((len(vocab), emb_dim))
    for i, word in enumerate(vocab):
        try:
            embedding_matrix[i] = model[word]
        except KeyError:
            # 对于词向量表中没有的词，使用随机初始化
            embedding_matrix[i] = np.random.normal(scale=0.1, size=(emb_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float32)

class Trainer:
    def __init__(self,
                 device,
                 model,
                 ckpt_path='./results/model.pt',
                 log_path='./results/train.log') -> None:
        self.device = device
        self.model = model.to(device)
        self.ckpt_path = ckpt_path
        self.train_loss_history = []
        self.valid_loss_history = []

        os.makedirs('./results', exist_ok=True) # exist_ok=True使得路径已存在时不会报错
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', # 格式
                            datefmt='%Y-%m-%d %H:%M:%S', # 时间信息
                            filename=log_path) # 日志位置和文件
        logging.info(f"Model saved to {self.ckpt_path}")

    def save_ckpt(self, best_epoch, optimizer, best_loss=float('inf')): # 保存模型，并保存中断时的训练状态
        model = self.model
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.ckpt_path}")
        torch.save({'epoch': best_epoch,
                    'best_loss':best_loss,
                    'model_state_dict': model.state_dict(),
                    'train_loss_history': self.train_loss_history,
                    'valid_loss_history': self.valid_loss_history,
                    'optimizer_state_dict': optimizer.state_dict()

        }, self.ckpt_path)

    def load_ckpt(self, optimizer): # 加载模型，返回上次训练停下时的epoch
        if os.path.exists(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_loss_history = checkpoint['train_loss_history']
            self.valid_loss_history = checkpoint['valid_loss_history']
            logging.info(f"Model loaded from {self.ckpt_path}")
            return checkpoint['epoch'], checkpoint['best_loss']
        else:
            return 0, 0

    def run_epoch(self, dataloader, optimizer, loss_func, epoch): # 训练一个epoch
        model = self.model
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader)) #用tqdm来生成迭代对象，在终端运行时就有实时进度条了
        d_loss, d_n = 0.0, 0 # 总loss和总个数，用于求本轮训练的平均loss

        for it, (X, Y) in pbar:
            X = X.to(self.device)
            Y = Y.to(self.device)
            outputs_idx, outputs = model(X, Y, teacher_forcing=True) # 使用teacher forcing
            outputs =  outputs.to(self.device)
            # print(f'outputs_idx shape: {outputs_idx.shape}')
            # print(f'outputs shape: {outputs.shape}')
            loss = 0.0
            for word_idx in range(outputs.size(1)):
                lo = loss_func(outputs[:, word_idx, :], Y[:, word_idx])
                if torch.isnan(lo) or torch.isinf(lo):
                    continue
                loss += lo
            loss = loss / outputs.size(1)
            d_loss += loss.item()
            d_n += X.shape[0]
            loss.backward()
            optimizer.step() # 更新权重
            optimizer.zero_grad() # zero_grad必须放在step后面！！！否则无法更新权重
            pbar.set_description(f"epoch{epoch + 1} iter {it}: loss = {loss.item():.5f}. lr = {optimizer.param_groups[0]['lr']:.7f}")

        train_loss = d_loss / d_n
        self.train_loss_history.append(train_loss)
        logging.info(f"Train, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.") # 记录日志
        return train_loss

    def validate(self, dataloader, loss_func): # 在验证集上计算loss和acc
        if dataloader is None:
            return None
        model = self.model
        model.eval()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        d_loss, d_n = 0.0, 0

        with torch.no_grad():
            for it, (X, Y) in pbar:
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs_idx, outputs = model(X, Y, teacher_forcing=True)  # 使用teacher forcing
                outputs = outputs.to(self.device)
                loss = 0.0
                for word_idx in range(outputs.size(1)):
                    lo = loss_func(outputs[:, word_idx, :], Y[:, word_idx])
                    if torch.isnan(lo) or torch.isinf(lo):
                        continue
                    loss += lo
                loss = loss / outputs.size(1)
                d_loss += loss.item()
                d_n += X.shape[0]
                pbar.set_description("Validation")

        val_loss = d_loss / d_n if d_n != 0 else 0
        self.valid_loss_history.append(val_loss)
        pbar.set_description(f"Validation, loss {val_loss:.5f}")
        logging.info(f"Validation, loss {val_loss:.5f}")
        return val_loss


    def train(self, train_dataloader,val_dataloader, optimizer, loss_func, max_epoch=10): # 训练模型
        start_epoch, best_loss = self.load_ckpt(optimizer)
        best_loss = self.valid_loss_history[-1] if self.valid_loss_history else float('inf')

        for epoch in range(start_epoch, max_epoch):
            train_loss = self.run_epoch(train_dataloader, optimizer, loss_func, epoch)
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader, loss_func)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    self.save_ckpt(best_epoch + 1, optimizer, best_loss)
            else: # 如果没有验证集，就用训练集的loss来判断
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_epoch = epoch
                    self.save_ckpt(best_epoch + 1, optimizer, best_loss)

        plt.plot(self.train_loss_history)
        if val_dataloader is not None:
            plt.plot(self.valid_loss_history)
            plt.legend(['train', 'valid'])
        else:
            plt.legend(['train'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss history')
        plt.savefig('results/train_loss.png')

    def test(self, data):
        pass

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device:{device}')

    # 测试Seq2Seq模型
    model = Seq2Seq(100, 100, 256, 512, 2)

    X = torch.randint(0, 100, (64, 20))
    Y = torch.randint(0, 100, (64, 10))
    print(f'X shape: {X.shape}')
    output_idx, output = model(X, Y)
    print(f'output_idx shape: {output_idx.shape}')
    print(f'output shape: {output.shape}')
    eng = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    output_idx, output = model.translate(eng)
    print(f'output shape: {output.shape}')
    print(f'output_idx shape: {output_idx.shape}')

    # 将fasttext词向量降维并保存
    fasttext.util.download_model('en', if_exists='ignore')
    fasttext.util.download_model('zh', if_exists='ignore')
    ch_vocab = load_vocab('./data/chinese_vocab.pkl')
    ch_model = fasttext.load_model('cc.zh.300.bin')
    fasttext.util.reduce_model(ch_model, 128)
    ch_model.save_model('cc.zh.128.bin')
    en_vocab = load_vocab('./data/english_vocab.pkl')
    en_model = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(en_model, 128)
    en_model.save_model('cc.en.128.bin')
    print('model saved')
