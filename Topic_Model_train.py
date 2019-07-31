from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)

# 读取数据
courses = []
courses_name = []
with open("coursera_corpus", "r", encoding="utf-8") as f:
    for line in f.readlines():
        courses.append(line.strip())
        courses_name.append(line.split("\t")[0])
# 预处理 小写 分词 去停词 去标点 提取词干 去除低频词
texts_lower = [[word for word in document.lower().split()] for document in courses]

texts_tokenized = [[word.lower() for word in word_tokenize(document)]
                   for document in courses]

english_stopwords = stopwords.words("english")
text_filtered_stopwords = [[word for word in document if word not in english_stopwords]
                           for document in texts_tokenized]

punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
text_filtered = [[word for word in document if word not in punctuations]
                 for document in text_filtered_stopwords]

st = LancasterStemmer()
text_stemmed = [[st.stem(word) for word in document] for document in text_filtered]

all_stems = sum(text_stemmed, [])
stem_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stem_once] for text in text_stemmed]


# 将文档的token映射为id
dictionary = corpora.Dictionary(texts)
# 将用字符串表示的文档转换为用id表示的文档向量
corpus = [dictionary.doc2bow(text) for text in texts]
# 基于用id表示的文档向量训练TF-IDF模型
tfidf = models.TfidfModel(corpus)
# 将用id表示的文档向量表示为用tf-idf值表示的文档向量
corpus_tfidf = tfidf[corpus]
# 基于tf-idf值表示的文档向量训练LSI模型
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
# 建立索引用于计算文档相似度
index = similarities.MatrixSimilarity(lsi[corpus])

ml_course = texts[210]  # 目标文档
ml_bow = dictionary.doc2bow(ml_course)
ml_lsi = lsi[ml_bow]
sims = index[ml_lsi]

sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sorted_sims[0:5])
