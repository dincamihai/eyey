import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def clean_content(content):
    return content.rstrip(string.punctuation).lower()


def get_content(message):
    if message.is_multipart():
        parts = message.get_payload()
        content = ''.join(part.get_payload(decode=True) for part in parts)
    else:
        content = message.get_payload(decode=True)
    return clean_content(content)


def get_words(tokens, porter, w2v_model):
    words = []
    unknown_words = []
    for token in tokens:
        if token.isalpha() and token not in stopwords.words('english'):
            try:
                words.append(
                    (porter.stem(token), w2v_model.get_vector(token))
                )
            except KeyError as kerr:
                unknown_words.append(token)
                continue
    return words, unknown_words


def get_features(words, label_fun):
    columns = ['stem'] + [i for i in range(50)]
    words_df = pd.DataFrame(
        [np.append(*row) for row in words], columns=columns).set_index('stem')
    words_counts = words_df.index.value_counts()
    words_df = words_df.merge(
        words_counts.to_frame(), left_index=True, right_index=True
    ).rename(
        columns={'stem': 'count'}
    ).drop_duplicates().sort_values(
        'count', ascending=False
    )
    top10_words_df = words_df[:10].reset_index().iloc[:, 1:-1]
    for i in range(10 - top10_words_df.shape[0]):
        top10_words_df = pd.concat([top10_words_df, pd.DataFrame([0.0 for i in range(50)]).T], axis=0)
    flat_df = top10_words_df.stack().reset_index().iloc[:, 2].to_frame().T
    flat_df['label'] = label_fun(flags) if label_fun else None
    assert flat_df.shape == (1, 501)
    flat_df.columns = ["w%s" % i for i in range(500)] + ['label']
    return flat_df


def process_content(content, porter, w2v_model, label_fun):
    try:
        tokens = word_tokenize(content)
    except Exception as exc:
        print(exc)
        return None, []

    words, unknown_words = get_words(tokens, porter, w2v_model)

    return get_features(words, label_fun), unknown_words


def process_message(message, porter, w2v_model, label_fun=None):
    try:
        content = get_content(message)
    except Exception as exc:
        print(exc)
        features, unknown_words = (None, [])
    else:
        features, unknown_words = process_content(content, porter, w2v_model, label_fun)
    return (features, unknown_words)


def process_folder(w2v_model, porter, maildir, label_fun):
    for key, message in maildir.iteritems():
        yield process_message(message, porter, w2v_model, label_fun)
