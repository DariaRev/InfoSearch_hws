from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import pandas as pd

vectorizer = CountVectorizer(analyzer='word')

names = ['моника', 'мон', 'рэйчел', 'рейч','чендлер', 'чэндлер', 'чен', 'фиби', 'фибс', 'росс','джоуи', 'джои', 'джо']


def stats_matr(matr_ind): # Функция для ответа на статистические вопросы. Печатает ответы
    matrix_freq = np.asarray(matr_ind.sum(axis=0)).ravel().tolist()
    dict_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
    index_max = matrix_freq.index(max(matrix_freq)) # ищем индексы самых популярных и нет слов
    index_min = matrix_freq.index(min(matrix_freq))
    print('Statistics by matrix', '\n', '-'*30)
    print('The most frequent word: ', dict_matrix[0][index_max])
    print('The rarest word: ', dict_matrix[0][index_min])
    inds = []
    words_every = []
    names_all = vectorizer.get_feature_names()
    # находим слова, которые есть во всех документах
    matr_ind_t = matr_ind.transpose() # транспонируем матрицу, поскольку нам нужно смотреть на колонки исходной матрицы. Если в этой колонке будет 0, то слово не попало в какой-то документ. Транспонирование используется для удобства
    li_matr_ind_t = matr_ind_t.tolist()
    for i in range(len(li_matr_ind_t)):
        if 0 in li_matr_ind_t[i]:
            continue
        else:
            inds.append(i)
    for i in inds:
        words_every.append(names_all[i])
    print('Words, that are in every document: ', set(words_every))
    nums = {}
    # Смотрим на матрицу с именами и частотами, для каждого имени из списка находим его частоту
    for i in range(len(dict_matrix[0])):
        if dict_matrix[0][i] in names:
            nums[dict_matrix[0][i]] = int(dict_matrix[1][i])
    max_name = max(nums, key=nums.get)
    print('The most popular name: ', max_name)


def get_index(corpus):  # Функция, которая делает матрицу и словарь из списка строк = корпуса
    matrix = vectorizer.fit_transform(corpus)
    dict_ind = {}
    for elem in corpus:
        words = elem.split()
        for word in words:
            if word not in dict_ind.keys():
                dict_ind[word] = []
            dict_ind[word].append(corpus.index(elem))
    for key, vals in dict_ind.items():
        dict_ind[key] = set(vals)
    return matrix.toarray(), dict_ind


def do_indexing_stats(corpus): # общая функция для вызова остальных
    matr_ind, dict_ind = get_index(corpus)
    stat_matr = stats_matr(matr_ind)
    return stat_matr
