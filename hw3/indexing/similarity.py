from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np


def get_query_vector(query, vectorizer):
    vector = vectorizer.transform(query)
    return vector


def get_similarity(query_vector, matr_index, docs, ans):
    similarities = np.dot(matr_index, query_vector.T).toarray()
    documents_indexes = similarities.nonzero()[0]
    sims = similarities[similarities.nonzero()].ravel()
    sorted_indexes = np.argsort(sims)[::-1]
    rank = documents_indexes[sorted_indexes]
    print('The most similar questions and answers to your question:')
    for index in rank[:5]:
        print(ans[index], '\n', docs[index])
        print('-'*30)


