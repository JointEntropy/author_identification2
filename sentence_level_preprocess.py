from word_lstm_preprocess import Mystem, Normalizer, filter_chars
import nltk
from itertools import chain
import pandas as pd
import configs
from utils import save_obj
from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences
russian_tokenizer = nltk.load('data/russian.pickle')


def split2sentences(texts):
    return texts.apply(lambda x: russian_tokenizer.tokenize(x))


def pad_sentences_df(df, max_sent_count, max_words_count, words_tokenizer):
    padded_sentences = df['text'].apply(
        lambda x: x[:max_sent_count] + ['hello there' for i in range(max_sent_count - len(x))])
    sentences_raw = chain.from_iterable(padded_sentences)
    encoded_sentences = words_tokenizer.texts_to_sequences(sentences_raw)
    padded_articles = pad_sequences(encoded_sentences, maxlen=max_words_count, padding='post',
                                    truncating='post')
    return np.array(list(group2articles(padded_articles, [max_sent_count] * df.shape[0])))


def group2articles(sentences, counts):
    """
    Упаковывает по counts_i предложений из sentences в тексты.
    :param sentences: всего предложений
    :param counts: число предложений в тексте
    :return: генератор текстов
    """

    batch = []
    count_iter = iter(counts)
    article_len = next(count_iter)
    for i, s in enumerate(tqdm(sentences)):
        batch.append(s)
        if len(batch) == article_len:
            try:
                article_len = next(count_iter)
            except StopIteration:
                pass
            yield batch
            batch = []
# words_count = pd.Series(list(chain.from_iterable(splitted.apply(lambda text: [len(s.split()) for s in text]))))

# for chunk in range(4):
    # data = pd.read_csv(configs.HUGE_DATA_PATH + '/loveread_fantasy_dataset_{}.csv'.format(chunk))


if __name__ == '__main__':
    data = pd.read_csv(configs.HUGE_DATA_PATH + '/loveread_fantasy_dataset_0.csv')#, nrows=1000)

    data.loc[0] = 'Привет мир!'  # первая строка битая
    filtered_data = filter_chars(data['text'])
    splitted = split2sentences(filtered_data)
    sentence_counts = splitted.apply(len)

    sentences = pd.Series(list(chain.from_iterable(splitted)))
    nm = Normalizer()
    normalized_sentences = nm.normalize(sentences)
    normalized_texts = pd.Series(list(group2articles(normalized_sentences, sentence_counts)))

    save_obj(normalized_texts, configs.HUGE_DATA_PATH + '/normalized_loveread_fantasy_sentences')
    # data.to_csv(configs.HUGE_DATA_PATH + '/normalized_loveread_fantasy_0.csv')