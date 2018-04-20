"""
Методы для работы с dataset'ом.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import configs
import os
import json
import configs
from word_lstm_preprocess import simple_tokenizer, filter_chars
from utils import split_sequence

headers_csv_path = 'data/prepared_info.csv'


def inverse_ohe(ohe_outputs, ohe_encoder):
    return ohe_encoder.active_features_[ohe_outputs.argmax(axis=1)]



def header_filter(headers_df):
    df = headers_df
    # оставляем авторов, чье число произвдеений больше 100
    df = df[df['total_count'] > 100]
    # выкидываем перемеоды
    df = df[df['translator'].isnull()]

    # фильтруем по конкретной категории
    # df['cats'] = df['cats'].fillna('[]')
    # df['cats'] = df['cats'].apply(eval)

    return df


def fetch_text_from_headers(headers_df):
    texts = []
    for i, row in tqdm(headers_df.iterrows(), total=headers_df.shape[0]):
        file_ = os.path.join(configs.CORPUS_PATH, str(i) + '.json')
        with open(file_, 'rb') as f:
            fcontent = json.load(f)
            text = fcontent['content']
            texts.append(text)
    return pd.DataFrame({'text': texts, 'author': headers_df['author'].values})


def filter_by_len(df, low_threshold=250):
    token_lens = df['text'].apply(len)
    mask = (token_lens > low_threshold)
    df = df[mask]
    return df

#
# def filter_by_genre(headers_df, genre="Русская проза, малые формы"):
#     """
#     Есть <img src="https://cs9.pikabu.ru/post_img/2017/02/24/5/1487918509199879809.jpg" width=100px style="display:inline;">:
#
#         1. работать с русской прозой, ибо большие тексты легче классифицировать. Здесь нужно юзать ebmedding'и(учить самому, или брать готовые).
#         2. работать с русской поэзией, и юзать char-rnn, потому что в некоторых текстах вообще по 50 символов
#     **Update** (нашёлся третий стул):
#         > можно забить на категории и просто взять из выборки те тексты, которые нам нравятся по размеру.
#     :param df:
#     :param genre:
#     :return:
#     """
#     df = headers_df
#     df['cats'] = df['cats'].fillna('[]')
#     df['cats'] = df['cats'].apply(eval)
#     mask = df['cats'].apply(lambda _: (genre in _) \
#         if _ not in [None, nan] else False)
#     return df[mask]
#
#
# def delete_duplicates_by_hash(df):
#     """
#     Update:
#     каким-то образом в выборке всё таки оказались дублирующиеся тексты.
#     Вычистим их, взяв хэш от самого текста. Можно,конечно, и проще как-то так...:
#     hash(имя автора+имя текста+...)  но не будем искать лёгких путей).
#     :param df:
#     :return:
#     """
#     df = df.copy()
#     df['text_hash'] = df['text'].apply(hash)
#     hashes_counts = df['text_hash'].value_counts()
#     bad_hashes = set(hashes_counts[hashes_counts != 1].index)
#     good_hashes = set(hashes_counts[hashes_counts == 1].index)
#     selection = []
#     for i, row in tqdm(df.iterrows(), total=df.shape[0]):
#         hash_ = row['text_hash']
#         if hash_ in bad_hashes:
#             selection.append(i)
#             bad_hashes = bad_hashes - set([hash_])
#         elif hash_ in good_hashes:
#             selection.append(i)
#     data = df.loc[selection, :]
#     return data
#

def filter_by_samples_count(df, samples_threshold=10, verbose=1):
    """
    смотрим, чтобы не было совсем уж редких авторов
    :param df:
    :param samples_threshold:
    :param verbose:
    :return:
    """
    counts = df.author.value_counts()
    data = df[df.author.isin(counts[counts.values > samples_threshold].index)]
    if verbose:
        print('Осталось {} сэмплов и {} уникальных автора'.format(data.shape[0], data.author.value_counts().shape[0]))
    return data


def preprocessing(df, encode,
                  inputlen=50,  # input max sequence length
                  shuffle=False,
                  split=True,
                  ohe=None):
    """
    Берёт df  с полями text и author.
    Для каждого текста из df['text'] применяет метод encode. Полученную последовательность усекает
    или дополняет до размера inputlen.
    Для каждого автора из df['authors'] кодирует его с помощью уже обученного ohe(one hot encoder)'а.
    После этого случайно сортирует все тексты(вместе с их метками).
    :param df: dataframe  с полями 'author' и 'text'
    :param encode: метод для получения из текстов нужной последовательности.
                            для char это метод разбивающий список текстов на символы и кодирующий их, а
                            для words это метод разбивающий список текстов на слова и кодирующий их.

    :param inputlen:  максимальная длина последовательности
    :param shuffle:   перемешивать ли смэплы
    :param split: если включен, то df число сэмплов  в обучение будет отличаться от кол-ва сэмплов в df.
    :param ohe:       обученный one hot encoder для авторов.
    :return: contexts,labels,[groups if split]
    """
    from keras.preprocessing.sequence import pad_sequences

    if ohe is None:
        labels = df['author'].values.reshape(-1, 1)
    else:
        labels = ohe.transform(df['author'].values.reshape(-1, 1))
    contexts, labels = encode(df['text'].values),  labels
    if split:
        def split_long(texts, class_labels, threshold):
            res_texts = []
            res_labels = []
            res_groups = []
            for i, (text, label) in enumerate(zip(texts, class_labels)):
                if len(text) > threshold:
                    text_split = split_sequence(text, threshold)
                    res_texts.extend(text_split)
                    res_labels.extend([label]*len(text_split))
                    res_groups.extend([i]*len(text_split))
            return res_texts, res_labels, res_groups
        contexts, labels, groups = split_long(contexts, labels, threshold=inputlen)

    # последовательности будут дополнены или усечены до одного размера в любом случае.
    contexts = pad_sequences(contexts, maxlen=inputlen)

    if shuffle:
        indices = np.arange(contexts.shape[0])
        np.random.shuffle(indices)
        contexts = np.asarray(contexts)[indices]
        labels = np.asarray(labels)[indices]
        if split:
            groups = np.asarray(groups)[indices]
    return [contexts, labels] + ([groups] if split else [])


def harmonize_textsdf(df, inputlen):
    """
    Harmonize dataset texts in such way:
    to long texts is splitted into parts to preserve each text size(in tokens)  in range 1.5*inputlen.
    :param df:
    :param inputlen:
    :return:
    """
    to_remove = []
    res = pd.DataFrame({
        'author': [],
        'text': []
    })
    for i, sample in df.iterrows():
        sample, label = sample['text'], sample['author']
        if len(sample) > 1.5*inputlen:
            to_remove.append(i)
            split = split_sequence(sample, inputlen)
            extra_split = pd.DataFrame({'text': pd.Series(split).apply(lambda x: ''.join(x)), 'author': label})
            res = pd.concat([res, extra_split], axis=0).reset_index(drop=True)
    res = res.reset_index(drop=True)
    remain_idx = list(set(df.index.values) - set(to_remove))
    res = pd.concat([df.loc[remain_idx], res], axis=0).reset_index(drop=True)
    return res




if __name__ == '__main__':
    # открываем данные с заголовками текстов.
    headers_df = pd.read_csv(headers_csv_path, sep=',', quotechar='/', index_col='id')
    #headers_df = headers_df[:11000]
    print('фильтруем тексты по метаинформации о них в строках заголовков...')
    headers_df = header_filter(headers_df)
    print('подгружаем сами тексты из заголовков...')
    texts_df = fetch_text_from_headers(headers_df)
    texts_df = filter_by_len(texts_df, 1000)

    median = np.median(texts_df['text'].apply(len))
    print(median)

    texts_df = harmonize_textsdf(texts_df, median)
    texts_df.to_csv(configs.HUGE_DATA_PATH+'/dataset.csv', index=False)

    # texts_df = pd.read_csv(configs.HUGE_DATA_PATH+'dataset.csv')
    # texts_df = do_many_things(texts_df)
    # print('сохраняем сбрасывая индекс(чтобы был без пропусков...')
    # texts_df.to_csv('data/dataset.csv', index=False)
