from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities
import logging
import numpy as np
import re
# logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def train_lsi(data):

    # 预处理：变小写 分词 去停词 去标点 提取词干 去除低频词
    texts_tokenized = [[word.lower() for word in word_tokenize(document)]
                       for document in data]

    english_stopwords = stopwords.words("english")
    text_filtered_stopwords = [[word for word in document if word not in english_stopwords]
                               for document in texts_tokenized]

    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']',
                    '&', '!', '*', '@', '#', '$', '%', '§', "``", "''"
                    'A', 'B', 'C', 'D', 'E', 'F', 'G',
                    'H', 'I', 'J', 'K', 'L', 'M', 'N',
                    'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z'
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    text_filtered = [[word for word in document if word not in punctuations]
                     for document in text_filtered_stopwords]

    pattern = [r"\d+", r"\."]
    for document in text_filtered:
        for word in document:
            for item in pattern:
                re.sub(item, "", word)

    # text_reduce_numbers = [[re.sub(item, "", word) for item in pattern for word in document]
    #                        for document in text_filtered]
    wnl = WordNetLemmatizer()
    text_reduced = [[wnl.lemmatize(word) for word in document] for document in text_filtered]

    # all_stems = sum(text_reduced, [])
    # stem_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    # texts = [[stem for stem in text if stem not in stem_once] for text in text_reduced]

    # 将文档的token映射为id
    dictionary = corpora.Dictionary(text_reduced)
    # 将用字符串表示的文档转换为用id表示的文档向量
    corpus = [dictionary.doc2bow(text) for text in text_reduced]
    # 基于用id表示的文档向量训练TF-IDF模型
    tfidf = models.TfidfModel(corpus)
    # 将用id表示的文档向量表示为用tf-idf值表示的文档向量
    corpus_tfidf = tfidf[corpus]
    # 基于tf-idf值表示的文档向量训练LSI模型
    # lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    # lsi.print_topics(2)

    # corpus_lsi = lsi[corpus_tfidf]
    # for doc in corpus_lsi:
    #     print(doc)
    # 基于tf-idf表示的文档向量训练LDA模型
    num_topics = 50
    # lda = models.ldamodel.LdaModel
    ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=100)

    return ldamodel, dictionary, text_reduced


    # 建立索引用于计算文档相似度
    # index = similarities.MatrixSimilarity(lsi[corpus])

    # ml_course = texts[210]  # 目标文档
    # ml_bow = dictionary.doc2bow(ml_course)  # 目标文档转换为id表示的文档向量
    # ml_lsi = lsi[ml_bow]    # id表示的目标文档向量转换为lsi向量
    # sims = index[ml_lsi]    # 利用索引计算相似度

    # sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(sorted_sims[0:5])
    # tmp = ldamodel.print_topics(num_topics=num_topics, num_words=num_show_term)
    # for i in tmp:
    #     print(i)


def get_keywords(ldamodel, dictionary, train_text, key):
    # 获取每个主题的关键词
    '''
    算法思路：候选的关键词与抽取的主题计算相似度并进行排序，得到最终的关键词
    关键点：候选关键词和抽取的主题如何计算相似度？
    解决办法：  每个主题由N个单词*概率 的集合来代表。
                每个文本属于k个主题，把k个主题所包含的词赋予该文档，便得到每个文档的候选词关键词。
                如果文档分词后得到的词语在候选关键词中，那么将其作为关键词提取出来。
    '''
    num_topics = 50
    num_show_term = 20
    topic_word_dict = {}
    for topic_id in range(num_topics):
        templist = []
        term_distribute_all = ldamodel.get_topic_terms(topicid=topic_id)
        # a1 = lda.print_topic()
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        for t in term_id:
            templist.append(dictionary.id2token[t])
        topic_word_dict[topic_id] = templist

    # 判断训练集每个文档所属主题和其候选关键词
    doc_topic_dict = {}  # key:第i篇文档 value: 第i篇文章的主题分布
    doc_word_dict = {}   # key:第i篇文档 value: 第i篇文章的主题所包含的词语
    m = len(train_text)
    for i in range(m):
        templist2 = []
        test_doc = train_text[i]
        doc_bow = dictionary.doc2bow(test_doc)  # 将文档转换成id
        num_show_topic = 3  # 每个文档取其前3个主题
        doc_topics = ldamodel.get_document_topics(bow=doc_bow)  # 得到某文档的主题分布

        doc_topic_dict[i] = doc_topics[:num_show_topic]
        for topic in doc_topics:
            temp_word = topic_word_dict[topic[0]]
            templist2 += temp_word
        doc_word_dict[i] = templist2
    final_dict = {}
    for i in range(m):
        keyword = []
        for word in train_text[i]:
            # print(word)
            if word in doc_word_dict[i]:
                keyword.append(word)
            final_dict[key[i]] = list(set(keyword))

        # print("文本%d的关键词: " % (i+1), list(set(keyword)))
        # print("文本%d " % i, " ".join(texts[i]))
        # print("文本%d的主题: " % i)
        # print(doc_word_dict[i])

    return final_dict


def save_dict(dict_to_save):
    cnt = 0
    with open("0.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            if "".join(line).find("_id") != -1:
                temp_id = "".join(line).replace("_id: ", "")
                print(temp_id)
            with open("0_add_dict.txt", "a", encoding="utf-8") as f2:
                if line != "\n":
                    f2.write(line)
                    # f2.write("\n")
                else:
                    # cnt += 1
                    for key, val in dict_to_save.items():
                        if key == temp_id:
                            f2.write("keywords:" + str(val) + "\n")
                            f2.write("\n")
            if cnt == 10:
                break


def read():
    text = []
    key = []
    with open("0.txt", "r", encoding="utf-8") as f:
        for item in f.readlines():
            if item.find("text:") != -1:
                text.append(item.replace("text:", ""))
            if item.find("_id: ") != -1:
                key.append(item.replace("_id: ", ""))
            # if len(text) == 10:
                # break

    return text, key


# def save_txt(list_to_save):
#     with open("0_save.txt", "a", encoding="utf-8") as f:
#         for item in list_to_save:
#             f.write("".join(item) + "\n")


if __name__ == "__main__":
    # read_text()
    texts, final_key = read()
    # print(final_key)
    # save_txt(texts)
    lda_model, dictionary, trian_data = train_lsi(texts)
    final_dictionary = get_keywords(lda_model, dictionary, train_text=trian_data, key=final_key)
    save_dict(final_dictionary)
