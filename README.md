# InfoSearch_hws

Для уcтановки библиотек можно воспользоаться следующим кодом:
```python
python3 -m pip install -r requirements.txt
```

## Домашняя работа 1. Индексы

Эта работа посвящена исследованию обратных индексов и нахождению разных статистик. 
Использованные данные - [субтитры друзей](https://disk.yandex.ru/d/4wmU7R8JL-k_RA?w=1)

Директория ```indexing``` является пакетом с разными модулями.

Данный пакет подключается в главной программе main.py

Содержимое пакета: 
- модуль ```preprocess.py```, отвечающий за лемматизацию, приведение к нижнему регистру, удаление стоп-слов и пунктуации
- модуль ```index.py```, отвечающий за то, чтобы создать индекс в двух форматахх: матрица и словарь, а также вывести статистику по индексу
- ``` __init__.py```, который нужен для подключения модулей

В файле ```main.py``` я подключаю пакет indexing и работаю с функциями из этого пакета.

Для запуска можно использовать команду 
```python
>>> python3 main.py (path_to_directory)
```

## Домашняя работа 2. TF-IDF

Эта работа посвящена поиску similarity и TF-IDF. Данные используются те же, что в первом задании.
Все нужные файлы находятся в директории hw2.

Файлы: 
- ```main.py``` - главный файл программы, из которого идет запуск кода
- ```index.py``` - создает индекс в двух форматах - матрица и словарь
- ```preprocess.py``` - занимается препроцессингом
- ```similarity.py``` - делает вектор для заданного запроса, а также вычисляет similarity запроса и всех документов
- ``` __init__.py``` - для подключения модулей

 Usage:
1) Если корпус новый, и до этого для него не строился индекс
 ```
 python3 main.py search -p <path-to-data> <query>
```
2) Если до этого вычислялся индекс корпуса
 ```
 python3 main.py search <query> 
  ```
#### ВАЖНО! 
Для корректной работы второго метода нужно иметь файлы в текущей директории:

```files_names.txt``` - с названиями документов

```vectorizer.pk``` - с векторизатором данных

```sparse_matrix.npz``` - с индексом документов


## Домашняя работа 3. BM-25

Эта работа посвящена поиску similarity при помощи BM-25. Данные используются [отсюда](https://www.kaggle.com/datasets/bobazooba/thousands-of-questions-about-love?resource=download).
Все нужные файлы находятся в директории hw3.

### Для работы я использовала вопросы в качестве документов и ответы в качестве названий документов.
При вводе запроса пользователю будет выводиться вопрос, похожий на тот, что он задал, и ответ пользователя с самым большим значением value.
Всего пользователю выводится 5 самых популярных вопросов и ответов.

Файлы: 
- ```main.py``` - главный файл программы, из которого идет запуск кода
- ```index.py``` - создает индекс 
- ```preprocess.py``` - занимается препроцессингом
- ```similarity.py``` - делает вектор для заданного запроса, а также вычисляет similarity запроса и всех документов
- ``` __init__.py``` - для подключения модулей

 Usage:
1) Если корпус новый, и до этого для него не строился индекс
 ```
 python3 hw3/main.py search -p <path-to-data> <query>
```
2) Если до этого вычислялся индекс корпуса
 ```
 python3 hw3/main.py search <query> 
  ```
#### ВАЖНО! 
Для корректной работы второго метода нужно иметь файлы в текущей директории:

```files_names.txt``` - с названиями документов

```vectorizer.pk``` - с векторизатором данных

```sparse_matrix.npz``` - с индексом документов

```answers.txt``` - файл с ответами


## Домашняя работа 4. Bert

Эта работа посвящена поиску similarity при помощи Bert. Данные используются [отсюда](https://www.kaggle.com/datasets/bobazooba/thousands-of-questions-about-love?resource=download).
Все нужные файлы находятся в директории hw4.

Всего пользователю выводится 5 самых популярных вопросов и ответов.

Также можно найти, какая из моделей (bert или bm-25) лучше работает.

Файлы: 
- ```main.py``` - главный файл программы, из которого идет запуск кода
- ```index.py``` - создает индекс 
- ```preprocess.py``` - занимается препроцессингом
- ```similarity.py``` - делает вектор для заданного запроса, а также вычисляет similarity запроса и всех документов
- ``` __init__.py``` - для подключения модулей

 Usage:
1) Если корпус новый, и до этого для него не строился индекс
 ```
 python3 hw4/main.py search -p <path-to-data> -model (bm-25/bert) <query>
```
2) Если до этого вычислялся индекс корпуса
 ```
 python3 hw4/main.py search -model (bm-25/bert) <query> 
  ```
3) Если нужно посмотреть accuracy моделей
 ```
 python3 hw4/main.py search --acc <query>
  ```
  Или для нового корпуса
   ```
 python3 hw4/main.py search --acc -p <path-to-data> <query>
  ```
  
#### ВАЖНО! 
Для корректной работы метода c precomputed corpus нужно иметь файлы в текущей директории:

```files_names.txt``` - с названиями документов

```vectorizer.pk``` - с векторизатором данных

```sparse_matrix_ans.npz``` - с индексом документов

```answers.txt``` - файл с ответами

```tensor_ans.pt``` - для модели берт для документов

```tensor_q.pt``` - для модели берт для запросов


## Проект. Поисковик

Чтобы запустить проект -
  ```
  streamlit run main.py
  ```
    
Данные используются [отсюда](https://www.kaggle.com/datasets/bobazooba/thousands-of-questions-about-love?resource=download).
Все нужные файлы находятся в директории project.

Поскольку файл для берта слишком большой, он не помещается в гитхаб, скачать его можно [отсюда](https://drive.google.com/file/d/1ZF3FOqv3b1XmUZjMmITaSSk1aY-YsZmB/view?usp=sharing), а затем положить в папку files
