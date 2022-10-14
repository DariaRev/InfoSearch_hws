from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn


def get_similarity(query_vector, mat_ans):
    similarities = cosine_similarity(mat_ans, query_vector)
    return similarities


def get_result(similarities, ans, ques):
    documents_indexes = similarities.nonzero()[0]
    sims = similarities[similarities.nonzero()].ravel()
    sorted_indexes = np.argsort(sims)[::-1]
    rank = documents_indexes[sorted_indexes]
    print('The most similar questions and answers to your question:')
    for index in rank[:5]:
        print(ques[index], '\n', ans[index])
        print('-'*30)


