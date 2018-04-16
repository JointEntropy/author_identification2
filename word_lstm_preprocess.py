
import string
from tqdm import tqdm
from itertools import chain
from collections import Counter
from utils import save_obj
import nltk
import pymorphy2
import pandas as pd
import configs
from char_lstm_preprocess import filter_chars
import numpy as np


oc2upos = {  # OPENCORPORA(pymorphy2)  to UPoS
    'NOUN'	: 'NOUN',
    'ADJF'	: 'ADJ',
    'ADJS'	: 'ADJ',
    'COMP'	: 'ADJ',
    'VERB'	: 'VERB',
    'INFN'	: 'VERB',

    'PRTF'	: 'VERB',  # причастие (полное)	прочитавший, прочитанная
    'PRTS'	: 'VERB',  # причастие (краткое)	прочитана

    'GRND'	: 'VERB',  # деепричастие	прочитав, рассказывая
    'NUMR'	: 'NUN',   # числительное	три, пятьдесят

    'ADVB'	: 'ADV',   # наречие

    'NPRO'	: 'NOUN',  # местоимение-существительное	он
    'PRED'	: 'VERB',  # предикатив	некогда
    'PREP'	: 'PRT',   # предлог	в

    'CONJ'	: 'SCONJ',  # союзы
    'PRCL'	: 'PRT',   # частицы
    'INTJ'	: 'INTJ'       # междометие	ой
}

morph = pymorphy2.MorphAnalyzer()


def simple_tokenizer(line):
    return line.translate(str.maketrans("", "", string.punctuation)).lower().split()


class WordsTokenizer:
    def __init__(self, ignore_unknown=False):
        self.ignore_unknown = ignore_unknown

    def fit_on_words(self, ready_words):
        self.knowing_words = ready_words
        self.word_index = set(list(ready_words.keys()))

    def texts_to_sequences(self, sentences, debug=False):
        texts = []
        total_missed = 0
        for line in tqdm(sentences):
            words = simple_tokenizer(line)
            if self.ignore_unknown:
                tokens = [self.knowing_words[word] for word in words if word in self.knowing_words]
            else:
                tokens = [self.knowing_words.get(word, len(self.word_index) + 1) for word in words]
            total_missed += (np.array(tokens) == len(self.word_index) + 1).sum()
            texts.append(tokens)
        if debug:
            print(total_missed)
        return texts


def inverse_ohe(ohe_outputs, ohe_encoder):
    return ohe_encoder.active_features_[ohe_outputs.argmax(axis=1)]


def extract_word_pos(word):
    try:
        int(word)
        word = 'x' * len(word)
        return '{}_NUM'.format(word)
    except ValueError:
        pass
    parsed = morph.parse(word)[0]
    return '{}_{}'.format(parsed.normal_form, oc2upos.get(parsed.tag.POS, 'X'))


def extract_from(texts, extractor, tokenizer):
    for i, line in enumerate(tqdm(texts)):
        words = tokenizer(line)
        yield from (extractor(word) for word in words)


def get_normalized_pos_texts(ser):
    """
    Normalize each text word and add PoS tag o it(as in dict for embeddings) in df['text'].
    :param ser: tokenized texts
    :return:
    """
    for i, words in enumerate(tqdm(ser)):
        yield [extract_word_pos(word) for word in words]


if __name__ == '__main__':
    ### TO PREPROCESS
    # output_words_path = configs.WORDS_PATH + '/uPoS_words'
    #
    # data = pd.read_csv('data/dataset.csv')
    # filtered_data = filter_chars(data['text'])
    # gen = chain(extract_from(filtered_data,
    #                          extractor=extract_word_pos,
    #                          tokenizer=simple_tokenizer))
    # poswords = Counter(gen)
    # # хранить counter правильнее, потому что потом если что можно выкинуть слишком редкие слова. Иначе
    # # поиск этих слов в embedding'ах происходит непозволительно долго
    #
    # save_obj(poswords, output_words_path)
    # print(poswords)



    ### TO PREPARE(normalize) DATASET TEXTS
    output_normalized_path = 'data/dataset'

    data = pd.read_csv('data/dataset.csv')
    filtered_data = filter_chars(data['text'])
    data['text'] = normalize_texts(data['text'])
    pd.data



# stop words appendix
"""
# import nltk
# #nltk.FreqDist() # фигня для подсчёта количества
# from nltk.corpus import stopwords
# stop = set(stopwords.words('russian'))
# len(stop)
"""

# appendix
"""
# import nltk
# import pymorphy2
# from nltk.stem.snowball import RussianStemmer
#
# from nltk.tokenize import RegexpTokenizer
#
# base_tokenizer = nltk.word_tokenize
# regexp_tokenizer = RegexpTokenizer(r'\w+').tokenize
#
# # Выбираем лемматизатор
# lems = True
# if lems:
#     morph = pymorphy2.MorphAnalyzer()
#     lmtzr = lambda token: morph.parse(token)[0].normal_form
# else:
#     lmtzr = RussianStemmer().stem
#
#
# def prepare_sample(sentence, tokenizer=base_tokenizer):
#     for word in tokenizer(sentence.lower()):
#         yield lmtzr(word)
#
#
# test_sent = data['text'][0]
# # list(prepare_sample(test_sent))
# X = []
# for i, row in tqdm(data.iterrows(), total=data.shape[0]):
#     X.append(list(prepare_sample(row['text'])))
# print(len(X))

# сохраним всю эту хрень на диск
# TEMP_PATH = '/media/grigory/Data/DIPLOM_DATA'
# ifp = os.path.join(TEMP_PATH, 'tokenized_input.json') #input file
# ofp = os.path.join(TEMP_PATH, 'labels.json')  # output file
# dfp = os.path.join(TEMP_PATH, 'encoded_dict.pkl')  # most freq words dict file
# import json
# with open(ifp,'w') as f:
#     json.dump(X, f)
# print('Input saved!')
# with open(ofp,'w') as f:
#     json.dump(data['author'].values.tolist(), f)
# print('Labels saved!')
"""