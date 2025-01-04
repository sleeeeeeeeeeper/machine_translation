import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import jieba


# 读取英文文件
def read_english_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()  # 读取所有行
    lines = [line.strip() for line in lines if line.strip()]
    df_en = pd.DataFrame(lines, columns=['english'])
    return df_en

# 读取中文文件
def read_chinese_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()  # 读取所有行
    lines = [line.strip() for line in lines if line.strip()]
    df_en = pd.DataFrame(lines, columns=['chinese'])
    return df_en

# 删除标点符号
def remove_punc(sentence):
    return re.sub(r"[\/\.\:\;\»\«\\\¡\¿\!\?\^\`\~\#\$\&\%\)\(\<\>,\、\^\+\_\-\*\"\t\n+，。！？【】（）《》“”‘’；：]"," ",sentence)

# 删除数字
def remove_numbers(sentences):
    return re.sub(r'\d+', '', sentences)

# 英文文本预处理
def english_preprocessing(row):
    sentences = sent_tokenize(row)  # 分句
    new_sentences = []
    for sent in sentences:
        new_sent = re.sub(r'\(.*?\)', '', sent)  # 去括号及括号内内容
        new_sent = remove_punc(new_sent.lower())  # 转小写并去标点
        new_sent = remove_numbers(new_sent)  # 去数字
        new_sent = new_sent.split()  # 分词
        new_sentences += new_sent
    return new_sentences

# 中文文本预处理
def chinese_preprocessing(row):
    if not isinstance(row, str):
        row = str(row)
    sentences = row.split('。')  # 中文按句号分句
    new_sentences = []
    for sent in sentences:
        new_sent = re.sub(r'（.*?）', '', sent)  # 去括号及括号内内容
        new_sent = remove_punc(new_sent)  # 去标点
        new_sent = remove_numbers(new_sent)  # 去数字
        new_sent = jieba.cut(new_sent)  # 使用jieba分词
        new_sent = list(new_sent)
        # 去掉分词后多余的空格
        new_sent = [word for word in new_sent if word.strip()]  # 去除空字符串
        new_sentences += new_sent

    return new_sentences


# 主预处理函数
def preprocess(english_file_path, chinese_file_path):
    # 读取英文和中文文件
    df_en = read_english_file(english_file_path)
    df_zh = read_chinese_file(chinese_file_path)

    # 合并两个DataFrame
    df = pd.concat([df_en, df_zh], axis=1)  # 将英文和中文列合并

    # 对英文和中文列分别应用预处理
    df['english'] = df['english'].apply(english_preprocessing)
    df['chinese'] = df['chinese'].apply(chinese_preprocessing)

    # 计算英文和中文句子的长度
    df['eng_length'] = df['english'].apply(lambda t: len(t))
    df['ch_length'] = df['chinese'].apply(lambda t: len(t))

    # 去掉长度大于50的行
    df = df[(df['eng_length'] <= 50) & (df['ch_length'] <= 50)]

    df.to_csv('data/preprocessed_data.csv', index=False)  # 保存预处理后的数据

    return df

if __name__ == '__main__':
    nltk.download('punkt', download_dir='data')
    nltk.download('punkt_tab', download_dir='data')
    nltk.data.path.append('data')

    # 读取英文文件
    df_en = read_english_file('data/TED2020.en-zh_cn.en')
    print(df_en.head(5))

    # 读取中文文件
    df_zh = read_chinese_file('data/TED2020.en-zh_cn.zh_cn')
    print(df_zh.head(5))

    # 预处理
    df = preprocess('data/TED2020.en-zh_cn.en', 'data/TED2020.en-zh_cn.zh_cn')
    print(df.head(5))
