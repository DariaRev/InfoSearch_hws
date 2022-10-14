import re
import sys
import numpy as np
import json
import pickle as pickle
import scipy.sparse
import logging
import indexing
import os
import torch
from tqdm.auto import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logging.basicConfig(level=logging.INFO)

curr_dir = os.getcwd()
txt_files = []
# /Users/Dasha/Downloads/data.jsonl
def make_argparser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='subparser')
    search = sub.add_parser('search')
    search.add_argument('query', type=str, nargs='+')
    search.add_argument('-p', type=str, help='path to directory')
    search.add_argument('-model', type=str, help='Type "bert" for bert model or "bm-25" for bm-25 model', default="bm-25")
    search.add_argument('--acc', action='store_true')
    search.add_argument('--no-acc', dest='feature', action='store_false')
    search.set_defaults(acc=False)
    return parser


def go_through_files(curr_dir):  # функция для сбора всех файлов в один список
    txt_files = {}
    with open(curr_dir) as f:
        corpus = list(f)[:50000]
        for que in tqdm(corpus):
            all_inf = json.loads(que)
            quest = all_inf['question']
            answ = all_inf['answers']
            if answ:
                sorted_answers = [answer['author_rating']['value'] for answer in answ]
                a = answ[np.argmax(sorted_answers)]["text"]
            txt_files[a] = quest
    return txt_files


def preprocess(txt_files): # функция препроцессинга (для вызова модуля из пакета), на выходе - список строк после препроцессинга
    ans = []
    ques = []
    for file, text in tqdm(txt_files.items()):
        ans.append([' '.join(indexing.preprocess(file))])
        ques.append([' '.join(indexing.preprocess(text))])
    return ans, ques


def get_index_bm(ans, ques):
    vec, mat_ans = indexing.get_index(ans)
    vec2, mat_ques = indexing.get_index(ques)
    return vec, mat_ans, mat_ques


def get_index_bert(ans, ques):
    mat_ans = indexing.get_index_bert(ans)
    mat_ques = indexing.get_index_bert(ques)
    vec = ''
    return vec, mat_ans, mat_ques


def save_bm(mat_ans, mat_ques, vec):
    scipy.sparse.save_npz('sparse_matrix_ans.npz', mat_ans)
    scipy.sparse.save_npz('sparse_matrix_ques.npz', mat_ques)
    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vec, fin)


def save_bert(mat_ans, mat_ques):
    torch.save(mat_ans, 'tensor_ans.pt')
    torch.save(mat_ques, 'tensor_q.pt')


def load_bm():
    mat_ans = scipy.sparse.load_npz('sparse_matrix_ans.npz')
    mat_ques = scipy.sparse.load_npz('sparse_matrix_ques.npz')
    with open('vectorizer.pk', 'rb') as fin1:
        vec = pickle.load(fin1)
    return mat_ans, mat_ques, vec


def load_bert():
    mat_ans = torch.load('tensor_ans.pt', map_location=torch.device('cpu'))
    mat_ques = torch.load('tensor_q.pt', map_location=torch.device('cpu'))
    return mat_ans, mat_ques


def get_sim_bm( query_vector_new_1, mat):
    sim = indexing.get_similarity(query_vector_new_1, mat)
    return sim


def get_sim_bert(args_new, mat_ans):
    mat_ans_n = []
    for elem in mat_ans:
        mat_ans_n.append(elem.numpy().ravel())
    mat_ans_nn = np.asarray(mat_ans_n)
    query_vector_new_1 = indexing.get_query_vector_bert(args_new)
    q = query_vector_new_1.numpy()
    sim = indexing.get_similarity(q, mat_ans_nn)
    return sim


def make_files(arguments):
    txt_files = go_through_files(arguments.p)
    ans, ques = preprocess(txt_files)
    ans_file_names = list(txt_files.keys())
    ques_file = list(txt_files.values())
    if arguments.model == "bm-25":
        vec, mat_ans, mat_ques = get_index_bm(ans, ques)
    else:
        vec, mat_ans, mat_ques = get_index_bert(ans, ques)

    # save files for working without preprocessing corpus
    logging.info('Saving needed files')
    with open('answers.txt', 'w') as f_a:
        for elem in ans_file_names:
            f_a.write(elem)
            f_a.write('\n')
    with open('files_names.txt', 'w') as f:
        for q in ques_file:
            f.write(q)
            f.write('\n')

    if arguments.model == "bm-25":
        save_bm(mat_ans, mat_ques, vec)
    else:
        save_bert(mat_ans, mat_ques, vec)
    return ans, ques, mat_ans, mat_ques, vec


def read_files(arguments):
    try:
        logging.info('Trying to open files')
        with open('files_names.txt') as f1:
            ques = f1.read().split('\n')
        with open('answers.txt') as f_a_fin:
            ans = f_a_fin.read().split('\n')
        if arguments.model == "bm-25":
            mat_ans, mat_ques, vec = load_bm()
        else:
            mat_ans, mat_ques = load_bert()
            vec = ''
    except ValueError:
        raise ValueError('Invalid path to file or file does not exist')
    return ans, ques, mat_ans, mat_ques, vec


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


def accuracy_bm(ques):
    mat_ans_bm, mat_ques_bm, vec = load_bm()
    ques_new = [' '.join(indexing.preprocess(text)) for text in tqdm(ques)]
    mat_ques_bm = indexing.get_query_vector(ques_new, vec)
    mat_bm = eval_bm(mat_ans_bm[:10000], mat_ques_bm[:10000])
    top5_bm = compute_score(mat_bm)
    print("BM-25: ", top5_bm)


def accuracy_bert():
    mat_ans_bert, mat_ques_bert = load_bert()
    mat_ques_bert = mat_ques_bert[:10000]
    mat_ans_bert = mat_ans_bert[:10000]
    a_bert = eval_bert(mat_ans_bert)
    q_bert = eval_bert(mat_ques_bert)

    cos_sim = indexing.get_similarity(q_bert, a_bert)
    top5_bert = compute_score(cos_sim)
    print("Bert: ", top5_bert)

def main():  # главная функция, вызывающая все остальные
    parser = make_argparser()
    arguments = parser.parse_args()

    if arguments.subparser == 'search':
        if not arguments.acc:
            if arguments.p:
                ans, ques, mat_ans, mat_ques, vec = make_files(arguments)
            else:
                ans, ques, mat_ans, mat_ques, vec = read_files(arguments)

            logging.info("Let's find similarities")

            args_new_str = ' '.join(arguments.query)
            args_new = [''.join(indexing.lemmatize(args_new_str)[:-1])]

            if arguments.model == "bm-25":
                query_vector_new_1 = indexing.get_query_vector(args_new, vec)
                sim = get_sim_bm(query_vector_new_1, mat_ans)
            else:
                sim = get_sim_bert(args_new, mat_ans)
            indexing.get_result(sim, ans, ques)
        elif arguments.acc:
            if not arguments.p:
                ans, ques, mat_1, mat_2, vec_1 = read_files(arguments)
                # bm-25
                accuracy_bm(ques)
                # bert
                accuracy_bert()
            else:
                txt_files = go_through_files(arguments.p)
                ans, ques = preprocess(txt_files)
                vec, mat_ans_bm, mat_ques_bm = get_index_bm(ans, ques)
                vec, mat_ans_bert, mat_ques_bert = get_index_bert(ans, ques)
                save_bm(mat_ans_bm, mat_ques_bm, vec)
                save_bert(mat_ans_bert, mat_ques_bert, vec)
                # bm-25
                accuracy_bm(ques)
                # bert
                accuracy_bert()

    else:
        raise Exception('Invalid choice for search')


if __name__ == "__main__":
    main()
