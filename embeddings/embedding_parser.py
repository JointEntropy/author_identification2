"""
### Работа с готовыми Embedding'ами

Потенциальные траблы:

- многие embedding обучены без стоп-слов. В задаче определения авторства они скорее нужны. Помимо этого, удалена
и пунктуация, которая тоже может оказаться полезной.


### Токенизация
Ссылки по теме:
- [Простая токенизация с помощью nltk](http://igorshevchenko.ru/blog/entries/textrank)
- [Умная токенизация для русского языка](https://github.com/mithfin/Sentence-Splitter-for-Russian)


Вопросы и задачи:

- что делать с кавычками «» ?
    - выкинуть/игнорировать
    - оставить(надо писать свою регулярку или, возможно, добавлять везде отступы перед и после них)
- выкинуть чиселки
- выкинуть прямые ссылки на автора в тексте


готовые опции:

- https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
```python
from keras.preprocessing.text import Tokenizer
```
Проблемы с тем, что нет нормализации и настройки токенизации


Ссылки по теме:
- [Обучение собственных embedding'ов с помощью gensim либы](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)
- [gensim](https://becominghuman.ai/word-embeddings-with-gensim-68e6322afdca)
- [Обучение собственных embedding'ов (вместе с описанием подготовки)](https://www.quora.com/How-can-one-train-own-word-embeddings-like-word2vec)
- [Видосик про обучение](https://www.youtube.com/watch?time_continue=5445&v=U0LOSHY7U5Q)

Тренированный word2vec:
- [Обученные эмбеддинги](http://rusvectores.org/ru/models/)

Глянуть чат ODS, канал nlp. Много кейсов

"""
import string
from itertools import chain
import numpy as np
import  pandas as pd
from utils import load_obj, save_obj
import csv
import os
import configs


def read_emb(filename, max_words=10**5):
    words = []
    with open(filename,'r') as f:
        N, d = f.readline().split()
        N,d = int(N), int(d)
        emb_matr = np.zeros(shape=(max_words, d), dtype=np.float64)
        for i, line in enumerate(f):
            word, vec = line.split(' ',1)
            words.append(word)
            emb_matr[i] = np.fromstring(vec, dtype=float, sep=' ')
            if i==max_words-1: break
    return words, emb_matr


def select_remain(filename, remain):
    words = []
    idx = 0
    with open(filename,'r') as f:
        N, d = f.readline().split()
        N,d = int(N), int(d)
        emb_matr = np.zeros(shape=(len(remain), d), dtype=np.float64)
        for i, line in enumerate(f):
            word, vec = line.split(' ', 1)
            if word in remain:
                words.append(word)
                emb_matr[idx] = np.fromstring(vec, dtype=float, sep=' ')
                idx += 1
            if i % 10**4==0: print(i)
    return words, emb_matr[:len(words)]


def get_words(ser):
    ser = ser.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)).split())
    return set(chain.from_iterable(ser))


if __name__ == '__main__':
    os.chdir('..')
    emb_path = configs.EMBEDDINGS_PATH + '/ruwikiruscorpora_upos_skipgram_300_2_2018.vec'
    words_path = configs.WORDS_PATH + '/PoS_words_counter'
    output_emb_path = configs.EMBEDDINGS_PATH + '/my_plus_corpora_emb'
    output_words_path = configs.WORDS_PATH + '/my_plus_corpora_words'


    print(load_obj(words_path))



    # # извлекаем первые 10**5 слов из файла с эмбедингами
    # pop_words, emb_matr = read_emb(emb_path, 2*10**5)
    #
    # all_words = load_obj(words_path).values()
    # remain_seek_words = list(all_words - set(pop_words))
    #
    # extra_words, extra_emb = select_remain(emb_path, remain_seek_words)
    # print('Не нашлось слов', len(set(remain_seek_words) - set(extra_words)))
    # print('Нашлось дополнительных слов', len(extra_words))
    #
    # # Сохраняем результат
    # total_words = pop_words + extra_words
    # total_emb = np.concatenate([emb_matr, extra_emb])
    # total_words = dict((word, idx) for idx, word in enumerate(total_words))
    #
    # save_obj(total_words, output_words_path)
    # save_obj(total_emb, output_emb_path)