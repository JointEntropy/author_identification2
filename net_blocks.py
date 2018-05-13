from extra_layers import AttentionWithContext
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, concatenate, GRU, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, CuDNNLSTM, Bidirectional, Dropout, CuDNNGRU, Reshape, \
    Conv2D, MaxPool2D, Concatenate, Flatten, regularizers, TimeDistributed, GlobalMaxPooling2D, subtract, \
    BatchNormalization, GlobalAveragePooling3D, GlobalAveragePooling2D, Activation, dot, Multiply, multiply


char_emb_dim = 150
char_hidden_size = 300


# def text_cnn():
#     count_symbols = ALPHABET_LEN
#     max_sequence_length = MAX_TEXT_CHARS
#     vector_size = char_emb_dim
#     filter_sizes = (3, 4, 5)
#     num_filters = max_sequence_length
#
#     inp = Input(shape=(max_sequence_length,))
#     emb = Embedding(count_symbols,
#                     vector_size,
#                     input_length=max_sequence_length,
#                     trainable=True)(inp)
#     dr = SpatialDropout1D(0.7)(emb)
#     reshape = Reshape((max_sequence_length, vector_size, 1))(dr)
#     conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], vector_size), padding='valid',
#                     kernel_initializer='normal', activation='relu')(reshape)
#     conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], vector_size), padding='valid',
#                     kernel_initializer='normal', activation='relu')(reshape)
#     conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], vector_size), padding='valid',
#                     kernel_initializer='normal', activation='relu')(reshape)
#
#     maxpool_0 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1),
#                           padding='valid')(conv_0)
#     maxpool_1 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1),
#                           padding='valid')(conv_1)
#     maxpool_2 = MaxPool2D(pool_size=(max_sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1),
#                           padding='valid')(conv_2)
#
#     concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
#     out = Flatten()(concatenated_tensor)
#
#     model = Model(inp, out)
#     return model


def chars_encoder(ALPHABET_LEN, MAX_TEXT_CHARS):
    """
    Encoder для символьного представления
    """
    encoder = Sequential()
    encoder.add(Embedding(output_dim=char_emb_dim,
                          input_dim=ALPHABET_LEN,
                          input_length=MAX_TEXT_CHARS,
                          mask_zero=False,
                          trainable=True))
    encoder.add(Dropout(0.15))
    encoder.add(Dense(1000))
    encoder.add(Bidirectional(CuDNNLSTM(
        units=char_hidden_size,
        return_sequences=True)))
    encoder.add(Bidirectional(CuDNNLSTM(units=char_hidden_size,
                                        return_sequences=True)))
    encoder.add(AttentionWithContext())
    return encoder


word_emb_dim = 300
word_hidden_size = 300


def words_encoder(MAX_NB_WORDS, MAX_TEXT_WORDS, emb_weights=None):
    """
    Encoder для представления в виде слов.
    """
    encoder = Sequential()
    encoder.add(Embedding(output_dim=word_emb_dim,
                          input_dim=MAX_NB_WORDS,
                          input_length=MAX_TEXT_WORDS,
                          weights=[emb_weights],
                          mask_zero=False,
                          name='word_emb',
                          trainable=False))
    encoder.add(Dropout(0.15))
    encoder.add(Bidirectional(CuDNNLSTM(units=word_hidden_size,
                                        return_sequences=True)))
    # encoder.add(BatchNormalization())
    # ncoder.add(CuDNNGRU(units=hidden_size, return_sequences=True))
    encoder.add(AttentionWithContext())

    return encoder


def get_classifier(emb,
                   MAX_TEXT_WORDS,
                   MAX_TEXT_CHARS,
                   MAX_NB_WORDS,
                   ALPHABET_LEN,
                   char_branch=False,
                   word_branch=True,
                   n_classes=64):
    if word_branch:
        word_inp = Input(shape=(MAX_TEXT_WORDS,), dtype='int32')
        word_encoded = words_encoder(MAX_NB_WORDS=MAX_NB_WORDS,
                                     MAX_TEXT_WORDS=MAX_TEXT_WORDS,
                                     emb_weights=emb)(word_inp)
        inp = word_inp
        branch = word_encoded
    if char_branch:
        char_inp = Input(shape=(MAX_TEXT_CHARS,), dtype='int32')
        # char_encoded = text_cnn()(char_inp) #
        char_encoded = chars_encoder(ALPHABET_LEN=ALPHABET_LEN, MAX_TEXT_CHARS=MAX_TEXT_CHARS)(char_inp)
        inp = char_inp
        branch = char_encoded

    if char_branch & word_branch:
        concatenated = Multiply()([word_encoded, char_encoded])
        out = Dense(n_classes, activation="softmax")(concatenated)
        return Model([word_inp, char_inp], out)
    else:
        drop = Dropout(0.15)(branch)
        compressed = Dense(200)(drop)
        out = Dense(n_classes, activation="softmax")(compressed)
        return Model(inp, out)
