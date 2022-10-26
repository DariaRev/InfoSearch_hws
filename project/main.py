import re
import sys
import numpy as np
import json
from time import time
import streamlit as st
import pickle as pickle
import scipy.sparse
import logging
import indexing
import os
import base64
import torch
from tqdm.auto import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logging.basicConfig(level=logging.INFO)

@st.cache(allow_output_mutation=True)
def load_data():
    mat_ans_bm = scipy.sparse.load_npz('files/sparse_matrix_ans_bm.npz')
    with open('files/vectorizer_bm.pk', 'rb') as fin1:
        vec_bm = pickle.load(fin1)
    logging.info('Downloaded files')
    mat_ans_bert = torch.load('files/tensor_ans.pt', map_location=torch.device('cpu'))
    with open('files/vectorizer_tfidf.pk', 'rb') as fin1:
        vec_tf = pickle.load(fin1)
    mat_ans_tf = scipy.sparse.load_npz('files/sparse_matrix_tfidf.npz')
    with open('files/answers.txt') as f_a_fin:
        ans = f_a_fin.read().split('\n')
    return mat_ans_bm,  vec_bm, mat_ans_bert, vec_tf, mat_ans_tf, ans


def get_sim_bert(args_new, mat_ans):
    mat_ans_n = []
    for elem in mat_ans:
        mat_ans_n.append(elem.numpy().ravel())
    mat_ans_nn = np.asarray(mat_ans_n)
    query_vector_new_1 = indexing.get_query_vector_bert(args_new)
    logging.info('Got query vector')
    q = query_vector_new_1.numpy()
    sim = indexing.get_similarity(q, mat_ans_nn)
    return sim


def eval_bm(corpus, query):
    sim = np.dot(query, corpus.T).toarray()
    return sim

def eval_bert(mat_ans):
    mat_ans_n = []
    for elem in mat_ans:
        mat_ans_n.append(elem.numpy().ravel())
    mat_ans_nn = np.asarray(mat_ans_n)
    return mat_ans_nn

def compute_score(matrix):
    sorted = np.argsort(matrix, axis=1)[:, :-6:-1]
    arr = np.arange(matrix.shape[0])
    q_range = np.expand_dims(arr, axis=1)
    q_res = np.sum(q_range == sorted, axis=1)
    comp = np.sum(q_res) / matrix.shape[0]
    return comp

def load_ques():
    with open('files/files_names.txt') as f_a_fin:
        ques = f_a_fin.read().split('\n')
    return ques


def main():  # главная функция, вызывающая все остальные
    @st.experimental_memo
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    img = get_img_as_base64("bg-image.jpg")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://img.freepik.com/free-vector/pastel-ombre-background-pink-purple_53876-120750.jpg?w=900&t=st=1666725564~exp=1666726164~hmac=970495e401fcb4d029e34a8848fa5644a5c547b7d2bf775a3a01bfd87dd351de");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    .stTextInput{{
    font-size: 20px;
    }}
    .stTextInput>div{{
    border-radius: 30px;
    margin-left: -5px;
    }}
    .button{{
    border-radius: 10px;
    }}
    </style>
    """
    mat_ans_bm, vec_bm, mat_ans_bert, vec_tf, mat_ans_tf, ans = load_data()
    st.markdown(page_bg_img, unsafe_allow_html=True)
    original_title = '<p style="font-family:Montserrat;color:#221441;font-size:32px;">Поговорим про любовь?...</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    metric = st.sidebar.selectbox("Выберите тип поиска", ["TF-IDF", "BM25", "BERT"])
    number = st.sidebar.slider('Количество выдаваемых ответов', 1, 300, 25)
    query = st.text_input('Напишите вашу любовную проблему')
    if st.button("Поиск"):
        if query:
            start_time = time()
            query_new = [' '.join(indexing.preprocess(query))]
            if metric == 'TF-IDF':
                query_vector = indexing.get_query_vector_bm(query_new, vec_tf)
                sim = indexing.get_sim_dot(query_vector, mat_ans_tf)
                res = indexing.sort_scores(sim, ans)
            elif metric == "BERT":
                logging.info('downloaded files')
                query_vector = indexing.get_query_vector_bert(query_new[0])
                stacked = torch.stack(mat_ans_bert)
                sim = indexing.get_similarity(query_vector, stacked)
                logging.info('Got similarity')
                res = indexing.sort_scores_bert(sim, ans)
            else:
                query_vector = indexing.get_query_vector_bm(query_new, vec_bm)
                logging.info('Got query vector')
                sim = indexing.get_sim_dot(query_vector, mat_ans_bm)
                logging.info('Got similarity')
                res = indexing.sort_scores(sim, ans)
            answ_time = f'<p style = "font-size: 20px; color: #3E8762">На поиск ответа ушло {round(time() - start_time, 2)}c.</p>'
            st.markdown(answ_time, unsafe_allow_html=True)
            for elem in res[:number]:
                st.write(elem)


if __name__ == "__main__":
    main()
