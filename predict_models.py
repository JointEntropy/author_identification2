from dataset import split_long_texts, preprocessing
from char_lstm_preprocess import filter_chars
from keras.models import load_model
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier


class PredictModel:
    def __init__(self, fp_model):
        self.fp_model = fp_model
        self.knn = KNeighborsClassifier(n_neighbors=5)

    def fit(self, texts, classes):
        self.knn.fit(texts, classes)

    def predict(self, X):
        features = self.predict_features(X)
        return self.knn.predict(features)

    def predict_features(self, X):
        X = self.fp_model.process(X)
        features = self.fp_model.model.predict(X)
        return features.mean(axis=0)


class BWordCharLSTM:
    def __init__(self, model_pth):
        model = load_model(model_pth)
        self.model = Model(inputs=model.input, outputs=model.layers[-2].output)

    def process(self, X):
        # нарезаем на куски, ибо кормить в сеть слишком большой не можем.
        split_threshold = self.model.split_threshold
        text_split = split_long_texts(X, split_threshold)  # теперь index - номер произведения, и он дублируется

        # предобработка на уровне слов.
        # извлекаем из модели параметры и использовавшиеся методы.
        nm = self.model.normalizer
        words_tokenizer = self.model.word_tokenizer
        MAX_TEXT_WORDS = self.model.params['MAX_TEXT_WORDS']

        # препроцессим вход.
        filtered_data = filter_chars(text_split)
        text_word = nm.normalize(filtered_data)
        text_word = preprocessing(text_word, encode=words_tokenizer.texts_to_sequences, inputlen=MAX_TEXT_WORDS)

        # предобработка на уровне символов
        ...

        return (text_word, text_char)
