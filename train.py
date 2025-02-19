import torch
from torch.utils.data import DataLoader, random_split
from dataset import load_data, collate_fn, load_vocab
from models import Seq2Seq, Trainer, load_fasttext_embeddings
import fasttext.util


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    train_data = load_data('ted', train=True)
    train_data, val_data = random_split(train_data, [int(train_data.__len__() * 0.8), int(train_data.__len__()) - int(train_data.__len__() * 0.8)], generator=torch.Generator().manual_seed(42))

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    ch_vocab = load_vocab('./data/chinese_vocab.pkl')
    en_vocab = load_vocab('./data/english_vocab.pkl')
    ch_model = fasttext.load_model('cc.zh.128.bin')
    en_model = fasttext.load_model('cc.en.128.bin')
    ch_embedding_matrix = load_fasttext_embeddings(ch_vocab, ch_model, emb_dim=128)
    en_embedding_matrix = load_fasttext_embeddings(en_vocab, en_model, emb_dim=128)
    model = Seq2Seq(8000, 8000, 128, 128, 4, zh_embeddings=ch_embedding_matrix, en_embeddings=en_embedding_matrix, max_len=50).to(device)
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.2) # ignore_index=2表示忽略填充词的损失

    trainer = Trainer(device, model, ckpt_path='./results/Seq2Seqmodel.pth', log_path='./results/Seq2Seqtrain.log')
    trainer.train(train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    max_epoch=10)
