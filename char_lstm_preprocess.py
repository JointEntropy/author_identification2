from tqdm import tqdm
from itertools import chain
import pandas as pd
from utils import save_obj
import configs

ALLOWED_CHARS = {chr(chr_idx) for chr_idx in range(ord('а'), ord('я')+1)}
ALLOWED_CHARS |= set('ё,.—?!: \t\n') #«»


def filter_chars(ser, allowed=ALLOWED_CHARS, forbidden2space=True):
    """
    Filter only allowed words.
    If forbidden2space is true, will replace forbidden chars into spaces, else just throw  forbidden chars away.
    from text. collection of texts.

    WARNING: As side effect will strip and lower inputs.
    :param ser:
    :param allowed: set of allowed chars.
    :param forbidden2space: if true, will replace forbidden chars into spaces, else just throw  them away.
    :return:
    """
    ser = ser.str.strip()
    ser = ser.str.lower()

    all_chars = set(chain.from_iterable(ser.values))
    forbidden = all_chars - allowed
    if forbidden2space:
        translation = str.maketrans(''.join(forbidden), ' ' * len(forbidden))
        return ser.str.translate(translation)
        # return ser.apply(lambda item: ''.join((c if c in allowed else ' ') for c in item))
    else:
        return ser.apply(lambda item: ''.join(filter(lambda ch: ch in allowed, item)))


class CharsTokenizer:
    def fit_on_mapping(self, mapping):
        self.mapping = mapping
        self.char_index = set(mapping.keys())

    def texts_to_sequences(self, sentences):
        texts = []
        for text in tqdm(sentences):
            text = list(self.mapping.get(ch, len(self.mapping) + 1) for ch in text)  # если встретился символ, которого нет
            # в словаре, то ставим ему свой вектор
            texts.append(text)
        return texts


if __name__ == '__main__':
    # открываем dataset
    output_dict_path = 'data/encode_char_mapping'

    data = pd.read_csv('data/dataset.csv')

    filtered_data = filter_chars(data['text'])
    # Assumed that texts chars already filtered
    alphabet = set(chain.from_iterable(filtered_data))
    print('Vocab size:', len(alphabet))
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # texts_repr = encode_chars(filtered_data,  alphabet)
    save_obj(char_to_int, output_dict_path)
    pass


