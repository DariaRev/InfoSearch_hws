from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
from tqdm.auto import tqdm


def get_index(corpus):  # Функция, которая делает матрицу из корпуса
    corpus = [' '.join(lemmas) for lemmas in list(corpus.values())]
    k = 2
    b = 0.75

    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(corpus)

    tfidf_vectorizer = TfidfVectorizer(norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()
    denom = (k * (1 - b + (b * len_d / avdl)))

    for i, j in tqdm(zip(*tf.nonzero())):
        tf[i, j] = (tf[i, j] * idf[j] * (k + 1)) / (tf[i, j] + denom[i])

    return vectorizer, tf
