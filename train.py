import torch
from torch.utils.data import DataLoader, random_split
from dataset import load_data, collate_fn
from models import Seq2Seq, Trainer, load_fasttext_embeddings
import fasttext.util


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    train_data = load_data('ted', train=True)
    train_data, val_data = random_split(train_data, [int(train_data.__len__() * 0.9), int(train_data.__len__()) - int(train_data.__len__() * 0.9)])
    test_data = load_data('ted', train=False)

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    ch_vocab = test_data.chinese_vocab
    en_vocab = test_data.english_vocab
    ch_model = fasttext.load_model('cc.zh.128.bin')
    en_model = fasttext.load_model('cc.en.128.bin')
    ch_embedding_matrix = load_fasttext_embeddings(ch_vocab, ch_model, emb_dim=128)
    en_embedding_matrix = load_fasttext_embeddings(en_vocab, en_model, emb_dim=128)
    model = Seq2Seq(8000, 8000, 128, 256, 2, zh_embeddings=ch_embedding_matrix, en_embeddings=en_embedding_matrix, max_len=50).to(device)
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.2) # ignore_index=2表示忽略填充词的损失

    trainer = Trainer(device, model, ckpt_path='./results/Seq2Seqmodel.pth', log_path='./results/Seq2Seqtrain.log')
    trainer.train(train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    max_epoch=20)
