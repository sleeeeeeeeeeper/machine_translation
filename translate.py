import streamlit as st
import fasttext
import fasttext.util
import torch
from dataset import load_vocab
from preprocess import remove_punc, remove_numbers
from models import Seq2Seq, load_fasttext_embeddings
import re


class Translater:
    def __init__(self, model, en_vocab, cn_vocab, device):
        self.model = model
        self.en_vocab = en_vocab
        self.id2english = {i: token for token, i in self.en_vocab.items()}
        self.cn_vocab = cn_vocab
        self.id2chinese = {i: token for token, i in self.cn_vocab.items()}
        self.device = device

    def translate(self, sentence, temperature=1.0):
        self.model.eval()
        sentence = re.sub(r'\(.*?\)', '', sentence)
        sentence = remove_punc(sentence.lower())
        sentence = remove_numbers(sentence)
        sentence = [self.en_vocab.get(token, 3) for token in sentence.split()]
        sentence = [0] + sentence + [1]  # 0: <sos>, 1: <eos>
        sentence = sentence + [2] * (self.model.max_len - len(sentence))  # 2: <pad>
        sentence = torch.tensor(sentence).unsqueeze(0).to(self.device)
        print(sentence)
        output_idx = self.model.translate(sentence, self.device, temperature)
        output_idx = output_idx.squeeze(0).numpy().tolist()
        output = [self.id2chinese[int(i)] for i in output_idx]
        output_str = ' '.join(output).replace(',', '').replace("'", '')
        return output_str

# 使用缓存加载词汇表
@st.cache_data
def load_vocab_cached(path):
    return load_vocab(path)

# 使用缓存加载 FastText 模型
@st.cache_resource
def load_fasttext_model_cached(path):
    return fasttext.load_model(path)

# 使用缓存加载模型权重
@st.cache_data
def load_model_weights_cached(path, weights_only, map_location):
    return torch.load(path, weights_only=weights_only, map_location=map_location)

def run_translation_app():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Device: {device}")
    ch_vocab = load_vocab_cached('./data/chinese_vocab.pkl')
    en_vocab = load_vocab_cached('./data/english_vocab.pkl')
    st.write("Vocab loaded.")
    ch_model = load_fasttext_model_cached('cc.zh.128.bin')
    en_model = load_fasttext_model_cached('cc.en.128.bin')
    ch_embedding_matrix = load_fasttext_embeddings(ch_vocab, ch_model, emb_dim=128)
    en_embedding_matrix = load_fasttext_embeddings(en_vocab, en_model, emb_dim=128)
    st.write("Embeddings loaded.")
    with st.spinner('Loading model...'):
        model = Seq2Seq(8000, 8000, 128, 256, 2, zh_embeddings=ch_embedding_matrix, en_embeddings=en_embedding_matrix,
                        max_len=50).to(device)
        model.load_state_dict(
            load_model_weights_cached('results/Seq2Seqmodel.pth', weights_only=True, map_location=torch.device(device))[
                'model_state_dict'])
    st.success('Model loaded!')
    translater = Translater(model, en_vocab, ch_vocab, device)

    st.title("English to Chinese Translator")
    input_text = st.text_area("English sentence:", "")
    output_text = st.empty()
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    if st.button("Translate"):
        if input_text.strip():
            translation = translater.translate(input_text, temperature)
            output_text.text_area("Chinese translation:", translation)
        else:
            st.write("Please enter a sentence to translate.")

if __name__ == '__main__':
    # 运行streamlit应用
    run_translation_app()

    # # 测试翻译器
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'device:{device}')
    # ch_vocab = load_vocab('./data/chinese_vocab.pkl')
    # en_vocab = load_vocab('./data/english_vocab.pkl')
    # ch_model = fasttext.load_model('cc.zh.128.bin')
    # en_model = fasttext.load_model('cc.en.128.bin')
    # ch_embedding_matrix = load_fasttext_embeddings(ch_vocab, ch_model, emb_dim=128)
    # en_embedding_matrix = load_fasttext_embeddings(en_vocab, en_model, emb_dim=128)
    # model = Seq2Seq(8000, 8000, 128, 256, 2, zh_embeddings=ch_embedding_matrix, en_embeddings=en_embedding_matrix, max_len=50).to(device)
    # print(model)
    # # model.load_state_dict(torch.load('results/Seq2Seqmodel.pth', weights_only=True)['model_state_dict'])
    # model.load_state_dict(torch.load('results/Seq2Seqmodel.pth', weights_only=True, map_location=torch.device('cpu'))['model_state_dict'])
    # translater = Translater(model, en_vocab, ch_vocab, device)
    # sentence = 'thank you for your help'
    # output = translater.translate(sentence)
    # print(output)
