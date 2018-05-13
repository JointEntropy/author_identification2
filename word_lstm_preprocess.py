import string
from tqdm import tqdm
from functools import partial
from itertools import chain
from collections import Counter
from utils import save_obj
import nltk
import pymorphy2
import pandas as pd
import configs
from char_lstm_preprocess import filter_chars
import numpy as np
from tqdm import tqdm

from pymystem3 import Mystem

# oc2upos = {  # OPENCORPORA(pymorphy2)  to UPoS
#     'NOUN'	: 'NOUN',
#     'ADJF'	: 'ADJ',
#     'ADJS'	: 'ADJ',
#     'COMP'	: 'ADJ',
#     'VERB'	: 'VERB',
#     'INFN'	: 'VERB',
#
#     'PRTF'	: 'VERB',  # причастие (полное)	прочитавший, прочитанная
#     'PRTS'	: 'VERB',  # причастие (краткое)	прочитана
#
#     'GRND'	: 'VERB',  # деепричастие	прочитав, рассказывая
#     'NUMR'	: 'NUN',   # числительное	три, пятьдесят
#
#     'ADVB'	: 'ADV',   # наречие
#
#     'NPRO'	: 'NOUN',  # местоимение-существительное	он
#     'PRED'	: 'VERB',  # предикатив	некогда
#     'PREP'	: 'PRT',   # предлог	в
#
#     'CONJ'	: 'SCONJ',  # союзы
#     'PRCL'	: 'PRT',   # частицы
#     'INTJ'	: 'INTJ'       # междометие	ой
# }


mystem2upos = {  # stolen from https://github.com/akutuzov/universal-pos-tags/blob/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map
    'A'    :   'ADJ',
    'ADV'   :  'ADV' ,
    'ADVPRO' : 'ADV',
    'ANUM'  :  'ADJ',
    'APRO'  :  'DET',
    'COM'   : 'ADJ',
    'CONJ'  :  'SCONJ',
    'INTJ'   : 'INTJ',
    'NONLEX' : 'X',
    'NUM'   :  'NUM',
    'PART'  :  'PART',
    'PR'    :  'ADP',
    'S'     :  'NOUN',
    'SPRO'  :  'PRON',
    'UNKN' :   'X',
    'V'    :   'VERB'
}


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
            words = line.split() #simple_tokenizer(line) ибо портит регистр и убирает подчёркивания
            if self.ignore_unknown:
                tokens = [self.knowing_words[word] for word in words if word in self.knowing_words]
            else:
                tokens = [self.knowing_words.get(word, len(self.word_index) + 1) for word in words]
            total_missed += (np.array(tokens) == len(self.word_index) + 1).sum()
            texts.append(tokens)
        if debug:
            print(total_missed)
        return texts


def extract_from(texts, extractor, tokenizer):
    for i, line in enumerate(tqdm(texts)):
        words = tokenizer(line)
        yield from (extractor(word) for word in words)


def pymorphy_normalizer(ser):
    """
    Normalize each text word and add PoS tag o it(as in dict for embeddings) in df['text'].
    :param ser: tokenized texts
    :return:
    """
    morph = pymorphy2.MorphAnalyzer()

    def extract_pos(word):
        try:
            int(word)
            word = 'x' * len(word)
            return '{}_NUM'.format(word)
        except ValueError:
            pass
        parsed = morph.parse(word)[0]
        return '{}_{}'.format(parsed.normal_form, oc2upos.get(parsed.tag.POS, 'X'))

    for i, words in enumerate(tqdm(ser)):
        yield [extract_pos(word) for word in words]


# for mystem
def pos_extractor(token, mapping):
    normal_form, pos = token['lex'], token['gr'].split('=')[0].split(',')[0]
    pos = mapping.get(pos, 'X') if mapping  is not None else pos
    return '{}_{}'.format(normal_form, pos)


def mystem_normalizer(texts, batch_size=150, mapping=mystem2upos):
    """
    Normalizer(lemmatisation and PoS tagging) with Mystem backend.
    :param texts:
    :param batch_size:
    :param mapping:
    :return:
    """
    m = Mystem()  # not very good place to store it.

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        total = ' $ '.join(batch.apply(lambda x: x.replace('\n', '').replace('$', '')))

        text = []
        for word in m.analyze(total):
            if word['text'] == '$':
                yield ' '.join(text)
                text = []
                continue
            try:
                token = word['analysis'][0]
            except (KeyError, IndexError) as e:
                continue
            text.append(pos_extractor(token, mapping=mapping))
        yield ' '.join(text)


class Normalizer:
    def __init__(self, backend='mystem', tokenizer=None):
        self.backend = backend
        self.tokenizer = tokenizer

    def normalize(self, texts):
        result = []
        if self.backend == 'mystem':
            # mystem expect text data not tokenized
            for text in tqdm(mystem_normalizer(texts), total=len(texts)):
                result.append(text)

        elif self.backend == 'pymorphy':
            if self.tokenizer:
                texts = self._tokenize(texts)
            for text in pymorphy_normalizer(texts):
                result.append(' '.join(text))
        else:
            raise ValueError('Invalid backend !')
        return result

    def _tokenize(self, texts):
        return self.tokenizer(texts)


if __name__ == '__main__':
    output_words_path = configs.WORDS_PATH + '/uPoS_words'

    data = pd.read_csv(configs.HUGE_DATA_PATH+'/normalized_loveread_fantasy_0.csv')
    data['text'] = data['text'].fillna('  ')
    # filtered_data = filter_chars(data['text'])
    filtered_data = data['text']
    gen = chain.from_iterable(t.split() for t in filtered_data)
    # gen = chain(extract_from(filtered_data,
    #                          extractor=partial(pos_extractor, mapping=mystem2upos),
    #                          tokenizer=simple_tokenizer))
    poswords = Counter(gen)
    # save_obj(poswords, output_words_path)
    print(poswords.most_common(10000)[2000:10000])

    #
    #
    # ### TO PREPARE(normalize) DATASET TEXTS
    # for chunk in range(4):
    #     data = pd.read_csv(configs.HUGE_DATA_PATH + '/loveread_fantasy_dataset_{}.csv'.format(chunk))
    #     filtered_data = filter_chars(data['text'])
    #     nm = Normalizer()
    #     data['text'] = nm.normalize(filtered_data)
    #     data.to_csv(configs.HUGE_DATA_PATH + '/normalized_loveread_fantasy_{}.csv'.format(chunk))



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
