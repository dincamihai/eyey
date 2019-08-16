import os
import re
import glob
import time
import email
import imaplib
import mailbox
import word2vec
import gensim
import pandas as pd
import argparse
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.python.saved_model import tag_constants
from nltk.stem.porter import PorterStemmer

from utils import get_messages, log, get_label
from config import get_connection
from params import PARAMS

from process_live import process_message
import constants



class Tagger(object):

    def __init__(self, outdir, folders):
        path = os.getcwd() + '/' + outdir + '/export/exporter/*'
        log.info("Loading model from: " + path)
        export_dir = glob.glob(path)[-1]
        self.predict_fn = predictor.from_saved_model(export_dir)
        self.porter = PorterStemmer()
        con = get_connection()
        ok, _ = con.list('INBOX')
        assert ok == "OK"
        # all_folders = [f.split()[-1] for f in folders]
        self.con = con
        self.folders = folders

    def predict(self, features):
        if features is None:
            return None
        inputs = features.to_csv(index=False, header=False).replace('\n', '')
        prediction = self.predict_fn({'csv_row': [inputs]})
        label = prediction['output'][0]
        # label = prediction['classes'][0][prediction['scores'][0].argmax()]
        # return label in ['1', b'1']
        return label == 1

    def next(self, readonly=True, search='NEW'):
        for folder in self.folders:
            self.con.select(folder, readonly=readonly)
            for (uid, msg, flags) in get_messages(self.con, folder, "UNSEEN %s" % search):
                self.con.store(uid, "-FLAGS", '\Seen')
                log.info('%s %s %s', folder, uid, msg['Subject'])
                yield uid, flags, process_message(msg, self.porter, None)
            for (uid, msg, flags) in get_messages(self.con, folder, "SEEN %s" % search):
                log.info('%s %s %s', folder, uid, msg['Subject'])
                yield uid, flags, process_message(msg, self.porter, None)

    def set_important(self, uid):
        self.con.store(uid, "-FLAGS", constants.NOT_IMPORTANT.decode('utf8'))
        self.con.store(uid, "+FLAGS", constants.IMPORTANT.decode('utf8'))
        log.info('%s %s', uid, constants.IMPORTANT)

    def set_notimportant(self, uid):
        self.con.store(uid, "-FLAGS", constants.IMPORTANT.decode('utf8'))
        self.con.store(uid, "+FLAGS", constants.NOT_IMPORTANT.decode('utf8'))
        log.info('%s %s', uid, constants.NOT_IMPORTANT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='default')
    args = parser.parse_args()
    unknown_words = []
    log.info('Checking for new email')
    # folders = ['INBOX', 'INBOX/suse-manager', 'INBOX/salt', ])
    tagger = Tagger(PARAMS[args.params]['outdir'], PARAMS[args.params]['folders'])

    # Train an messages marked with 'train' flag
    to_train = []

    for (uid, flags, (features, part_unknown_words)) in tagger.next(False, search='KEYWORD %s' % constants.RETRAIN.decode('utf8')):
        tagger.con.store(uid, "-FLAGS", constants.RETRAIN.decode('utf8'))
        if features is None:
            continue
        features.at[0, 'label'] = 1 if constants.SHOULD_BE_IMPORTANT in flags else 0
        to_train.append(features)

    if to_train:
        import model
        model.post_train_and_evaluate(pd.concat(to_train), './trained/', './data')
        # need to load the new trained model
        tagger = Tagger(PARAMS[args.params]['outdir'], PARAMS[args.params]['folders'])

    # Set flags
    for (uid, flags, (features, part_unknown_words)) in tagger.next(
        False,
        search="OR (%s %s) (%s)" % (
            'NOT KEYWORD %s' % constants.IMPORTANT.decode('utf8'),
            'NOT KEYWORD %s' % constants.NOT_IMPORTANT.decode('utf8'),
            'KEYWORD %s' % constants.RELABEL.decode('utf8'))
    ):
        tagger.con.store(uid, "-FLAGS", constants.RELABEL.decode('utf8'))
        important = tagger.predict(features)
        if important is None:
            log.warning('No features for %s', uid)
        elif important:
            tagger.set_important(uid)
        else:
            tagger.set_notimportant(uid)

    # for (uid, (features, part_unknown_words)) in tagger.next(False, search='UNSEEN'):
    #     if tagger.is_important(features):
    #         tagger.set_important(uid)
    #     else:
    #         tagger.set_notimportant(uid)
    #     tagger.con.store(uid, "-FLAGS", '\Seen')

    # for (uid, (features, part_unknown_words)) in tagger.next(False, search='SEEN'):
    #     if tagger.is_important(features):
    #         tagger.set_important(uid)
    #     else:
    #         tagger.set_notimportant(uid)
    #     unknown_words += part_unknown_words

    # if unknown_words:
    #     with open('data/unknown_words.csv', 'ab') as f:
    #         unkw_df = pd.DataFrame(
    #             unknown_words
    #         ).reset_index().groupby(0).count().sort_values(
    #             'index', ascending=False)
    #         f.write(unkw_df.to_csv(header=False).encode('utf8'))
