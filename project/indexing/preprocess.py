import nltk
import streamlit as st
from string import punctuation
from pymystem3 import Mystem
from nltk.corpus import stopwords
stopwords = stopwords.words('russian')

@st.cache(allow_output_mutation=True)
def download_all():
    nltk.download('stopwords')

download_all()
mystem = Mystem()
# Функция лемматизации текста, на выходе - лемматизированный текст в виде списка слов
def lemmatize(text):
    lemmas = mystem.lemmatize(text)
    return lemmas

# Функция избавления от пунктуации и стоп слов + низкий регистр у всех слов. На выходе - список отфильтрованных слов
def del_punct_stop(lemmas):
    text = [word.lower().strip(punctuation) for word in lemmas]
    spaces = ['', ' ', '\n']
    text = [word for word in text if word not in spaces]
    stop_words = set(stopwords)
    filtered_sentence = [w for w in text if not w.lower() in stop_words]
    return filtered_sentence

# функция препроцессинга, запускающая остальные. На выходе - чистый список слов
def preprocess(text):
    lemmas = lemmatize(text)
    all_prep = del_punct_stop(lemmas)
    return all_prep

