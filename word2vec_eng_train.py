# !/usr/bin/env python
# -*-coding: utf-8 -*-
import gensim.models.word2vec as word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import smart_open
import time
import re


def train_save_model(file_name):
    sentences = word2vec.Text8Corpus(file_name)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=2)
    saved_model_name = "".join(file_name.split(".")[:-1]) + '.model'
    model.save(saved_model_name)
    return saved_model_name


def load_model(model_name):
    model = word2vec.Word2Vec.load(model_name)
    print(model.wv.most_similar("united"))
    # print(model.wv.__getitem__(["document"]))


def txt_clean(file_name):
    # 读取并分词
    with smart_open.open(file_name, "r", encoding="utf-8") as text:
        words_list = word_tokenize("".join(text.readlines()))   # 英文分词工具
    print("start organizing data...")
    punctuation_list = [',', '.', ':', ';', '?', '(', ')', '[', ']', ' .',
                        '&', '!', '*', '@', '#', '$', '%', '-', '®', '\"',  '•', '§',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N',
                        'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z']   # 要去除的单个字符，标点符号
    # 去停词和标点符号
    filter_words = [word for word in words_list
                    if word not in list(set(stopwords.words("english")))
                    and word not in punctuation_list]
    print("Remove stop words done")
    # 词形还原
    wordnet_le = WordNetLemmatizer()
    lemmatize_words = [wordnet_le.lemmatize(word) for word in filter_words]
    print("Words Lemmatizer done")

    # 整理为字符串并输出
    cleaned_string = (" ".join(lemmatize_words)).lower()

    # 用正则表达式去除特殊字符
    pattern = ["\\d", "\\.", "§"]
    for i in range(len(pattern)):
        cleaned_string = re.sub(pattern[i], "", cleaned_string)

    out_cleaned_name = "".join(file_name.split(".")[:-1]) + '_' + 'cleaned.txt'  # 处理产生的文件名
    with smart_open.open(out_cleaned_name, "w", encoding="utf-8") as writer:
        writer.write(cleaned_string)
    print("Done")
    return out_cleaned_name


if __name__ == "__main__":
    fn = "acrobat-xi-pdf-accessibility-repair-workflow_decrypted_extracted.txt"
    cleaned_fn = "pdf_reference_17_extracted_cleaned.txt"
    model_name_ = "pdf_reference_17_extracted_cleaned.model"
    time_start = time.time()
    cleaned_file = txt_clean(fn)  # 数据预处理
    time_end = time.time()
    print("Total time: ", (time_end-time_start))
    # save_model = train_save_model(cleaned_file)   # 训练并保存模型
    # load_model("usc01@116-20_extracted_cleaned.model")  # 载入模型并计算
