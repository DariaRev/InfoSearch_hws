import re
import sys
import numpy as np
import json
import _pickle as pickle
import scipy.sparse
import logging
import indexing
import os
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
    preproc = {}
    for file, text in tqdm(txt_files.items()):
        preproc[file] = [' '.join(indexing.preprocess(text))]
    return preproc


def main():  # главная функция, вызывающая все остальные
    parser = make_argparser()
    arguments = parser.parse_args()

    if arguments.subparser == 'search':
        if arguments.p:
            logging.info('Get corpus and make preprocessing')
            txt_files = go_through_files(arguments.p)
            preproc = preprocess(txt_files)
            vec, mat = indexing.get_index(preproc)
            lii = list(preproc)
            ans = list(txt_files.values())

            # save files for working without preprocessing corpus
            logging.info('Saving needed files')
            with open('answers.txt', 'w') as f_a:
                for elem in ans:
                    f_a.write(elem)
                    f_a.write('\n')
            with open('files_names.txt', 'w') as f:
                f.write('\n'.join(list(preproc)))
            scipy.sparse.save_npz('sparse_matrix.npz', mat)
            with open('vectorizer.pk', 'wb') as fin:
                pickle.dump(vec, fin)
        else:
            try:
                logging.info('Trying to open files')
                with open('files_names.txt') as f1:
                    lii = f1.read().split('\n')
                with open('vectorizer.pk', 'rb') as fin1:
                    vec = pickle.load(fin1)
                with open('answers.txt') as f_a_fin:
                    ans = f_a_fin.read().split('\n')
                mat = scipy.sparse.load_npz('sparse_matrix.npz')
            except ValueError:
                raise ValueError('Invalid path to file or file does not exist')

        logging.info("Let's find similarities")
        args_new_str = ' '.join(arguments.query)
        args_new = [''.join(indexing.lemmatize(args_new_str)[:-1])]
        query_vector_new_1 = indexing.get_query_vector(args_new, vec)
        indexing.get_similarity(query_vector_new_1, mat, lii, ans)
    else:
        raise Exception('Invalid choice for search')


if __name__ == "__main__":
    main()
