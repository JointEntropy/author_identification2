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

headers_csv_path = 'data/prepared_info.csv'


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


def textsdf2dataset(df):
    pass


def filter_by_len(df, low_threshold=25):
    token_lens = df['text'].str.split().apply(len)
    mask = (token_lens > low_threshold)
    df = df[mask]
    return df


def filter_by_genre(headers_df, genre="Русская проза, малые формы"):
    """
    Есть <img src="https://cs9.pikabu.ru/post_img/2017/02/24/5/1487918509199879809.jpg" width=100px style="display:inline;">:

        1. работать с русской прозой, ибо большие тексты легче классифицировать. Здесь нужно юзать ebmedding'и(учить самому, или брать готовые).
        2. работать с русской поэзией, и юзать char-rnn, потому что в некоторых текстах вообще по 50 символов
    **Update** (нашёлся третий стул):
        > можно забить на категории и просто взять из выборки те тексты, которые нам нравятся по размеру.
    :param df:
    :param genre:
    :return:
    """
    df = headers_df
    df['cats'] = df['cats'].fillna('[]')
    df['cats'] = df['cats'].apply(eval)
    mask = df['cats'].apply(lambda _: (genre in _) \
        if _ not in [None, nan] else False)
    return df[mask]


def delete_duplicates_by_hash(df):
    """
    Update:
    каким-то образом в выборке всё таки оказались дублирующиеся тексты.
    Вычистим их, взяв хэш от самого текста. Можно,конечно, и проще как-то так...:
    hash(имя автора+имя текста+...)  но не будем искать лёгких путей).
    :param df:
    :return:
    """
    df = df.copy()
    df['text_hash'] = df['text'].apply(hash)
    hashes_counts = df['text_hash'].value_counts()
    bad_hashes = set(hashes_counts[hashes_counts != 1].index)
    good_hashes = set(hashes_counts[hashes_counts == 1].index)
    selection = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        hash_ = row['text_hash']
        if hash_ in bad_hashes:
            selection.append(i)
            bad_hashes = bad_hashes - set([hash_])
        elif hash_ in good_hashes:
            selection.append(i)
    data = df.loc[selection, :]
    return data
    pass


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


def preprocessing(df, tokenizer,
                  inputlen=50,  # input max sequence length
                  shuffle=False,
                  ohe=None):
    from keras.preprocessing.sequence import pad_sequences
    labels = ohe.transform(df['author'].values.reshape(-1, 1))
    contexts, labels = df['text'].values,  labels
    contexts = tokenizer.texts_to_sequences(contexts)
    contexts = pad_sequences(contexts, maxlen=inputlen)
    if shuffle:
        indices = np.arange(contexts.shape[0])
        np.random.shuffle(indices)
        contexts = np.asarray(contexts)[indices]
        labels = np.asarray(labels)[indices]
    return contexts,  labels


if __name__ == '__main__':
    headers_df = pd.read_csv(headers_csv_path, sep=',', quotechar='/', index_col='id')
    headers_df = header_filter(headers_df)
    texts_df = fetch_text_from_headers(headers_df)
    texts_df = filter_by_samples_count(texts_df)
    texts_df = filter_by_len(texts_df)
    texts_df.to_csv('data/dataset.csv', index=False)
