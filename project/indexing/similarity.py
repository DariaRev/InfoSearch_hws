#from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import pandas as pd
from torch.nn.functional import cosine_similarity


def get_similarity(query_vector, mat_ans):
    similarities = cosine_similarity(mat_ans, query_vector)
    return similarities


def get_sim_dot(query_vector, mat_ans):
    return mat_ans.dot(query_vector.T)


def sort_scores(sims, ans):
    sorted_scores = np.argsort(np.array(sims), axis=0)[::-1]
    docs_array = np.array(ans)
    sorted_docs = docs_array[sorted_scores.ravel()]
    return sorted_docs


def sort_scores_bert(sim, ans):
    sorted = torch.argsort(sim, descending=True)
    docs_array = np.array(ans)
    a = sorted[(sorted < 48099)]
    sorted_docs = docs_array[a.ravel()]

    return sorted_docs

def get_result(similarities):
    documents_indexes = similarities.nonzero()[0]
    sims = similarities[similarities.nonzero()].ravel()
    sorted_indexes = np.argsort(sims)[::-1]
    rank = documents_indexes[sorted_indexes]
    return rank


