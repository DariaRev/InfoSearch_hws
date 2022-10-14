from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os


def get_index(ans):  # Функция, которая делает матрицу из корпуса
    ans = [' '.join(a) for a in ans]
    k = 2
    b = 0.75

    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(ans)

    tfidf_vectorizer = TfidfVectorizer(norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(ans)
    idf = tfidf_vectorizer.idf_

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()
    denom = (k * (1 - b + (b * len_d / avdl)))

    for i, j in tqdm(zip(*tf.nonzero())):
        tf[i, j] = (tf[i, j] * idf[j] * (k + 1)) / (tf[i, j] + denom[i])

    return vectorizer, tf


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_index_bert(corpus):
    cuda = torch.device('cuda')
    corpus = [' '.join(a) for a in corpus]
    corpus_all = []
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device=cuda)
    for elem in tqdm(corpus):
        encoded_input = tokenizer(elem, padding=True, truncation=True, max_length=24, return_tensors='pt').to(device=cuda)
        with torch.no_grad():
            model_output = model(**encoded_input)
        corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        corpus_all.append(corpus_embeddings)
    return corpus_all


def get_query_vector(query, vectorizer):
    vector = vectorizer.transform(query)
    return vector


def get_query_vector_bert(query):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    encoded = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded)
    vector = mean_pooling(model_output, encoded['attention_mask'])
    return vector


