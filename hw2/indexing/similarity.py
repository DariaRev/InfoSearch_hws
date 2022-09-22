from sklearn.metrics.pairwise import cosine_similarity
import operator


def get_query_vector(query, vectorizer):
    vector = vectorizer.transform(query)
    return vector


def get_similarity(query_vector, matr_index, docs):
    similarities = list(cosine_similarity(query_vector, matr_index)[0])
    sim_dict = {k: v for k, v in zip(docs, similarities)}
    sorted_di = dict(sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True))
    print('Similarity in descending order:')
    for elem in sorted_di.keys():
        print(elem)
    print('The most similar to the query:')
    for key in list(sorted_di.keys())[:5]:
        print(key)

