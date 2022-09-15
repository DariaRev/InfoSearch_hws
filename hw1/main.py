import re
import sys
import indexing
import os
from tqdm.auto import tqdm
import argparse

curr_dir = os.getcwd()
txt_files = []


def go_through_files(curr_dir):  # функция для сбора всех файлов в один список
    for root, dirs, files in os.walk(curr_dir):
        for name in files:
            if re.match(r'Friends.*\.txt', name):
                txt_files.append(os.path.join(root, name))
    return txt_files


def preprocess(txt_files): # функция препроцессинга (для вызова модуля из пакета), на выходе - список строк после препроцессинга
    preproc = []
    for file in tqdm(txt_files):
        with open(file) as f:
            read = f.read()
            preproc.append(' '.join(indexing.preprocess(read)))
    return preproc


def main(data_dir):  # главная функция, вызывающая все остальные
    txt_files = go_through_files(data_dir)
    preproc = preprocess(txt_files)
    mat = indexing.do_indexing_stats(preproc)
    return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args(sys.argv[1:])
    if os.path.exists(args.data_dir):
        print("File exist")
    main(args.data_dir)
