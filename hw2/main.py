import re
import sys
import _pickle as pickle
import scipy.sparse
import logging
import indexing
import os
from tqdm.auto import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)

curr_dir = os.getcwd()
txt_files = []

def make_argparser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='subparser')
    search = sub.add_parser('search')
    search.add_argument('query', type=str, nargs='+')
    search.add_argument('-p', type=str, help='path to directory')
    return parser


def go_through_files(curr_dir):  # функция для сбора всех файлов в один список
    txt_files = {}
    for root, dirs, files in os.walk(curr_dir):
        for name in files:
            if re.match(r'.*\.txt', name):
                with open(os.path.join(root, name)) as f:
                    txt_files[name] = f.read()
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
            vec, mat = indexing.get_index(preproc, TfidfVectorizer)
            lii = list(preproc)
            # save files for working without preprocessing corpus
            logging.info('Saving needed files')
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
                mat = scipy.sparse.load_npz('sparse_matrix.npz')
            except ValueError:
                raise ValueError('Invalid path to file or file does not exist')

        logging.info("Let's find similarities")
        args_new = [' '.join(arguments.query)]
        query_vector_new_1 = indexing.get_query_vector(args_new, vec)
        similarity = indexing.get_similarity(query_vector_new_1, mat, lii)
    else:
        raise Exception('Invalid choice for search')



if __name__ == "__main__":
    main()
