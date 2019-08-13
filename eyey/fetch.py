import glob
import hashlib
import random
import shutil
import mailbox
import word2vec
import gensim
import argparse
from nltk.stem.porter import PorterStemmer
import pandas as pd

from utils import get_messages, log, get_label
from config import get_connection
from process_live import process_message


def clean_outputs():
    for f in glob.glob('data/*.csv'):
        try:
            shutil.os.remove(f) # start fresh each time
        except Exception as exc:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', default='INBOX', type=str, nargs='+')
    parser.add_argument('--csvdir', default='./data', type=str)
    args = parser.parse_args()
    clean_outputs()
    porter = PorterStemmer()
    unknown_words = []

    con = get_connection()

    for folder in args.folders:
        log.info(folder)
        con.select(folder, readonly=True)
        for (uid, msg, flags) in get_messages(con, folder, 'ALL'):
            log.info("%s %s", uid, flags)
            label = get_label(flags)
            if label is None:
                continue
            (features, message_unknown_words) =  process_message(msg, porter, label)
            split_hash = int(hashlib.md5(msg['subject'].encode('utf8')).hexdigest(), 16)
            mode = 'test' if (int(split_hash) % 10) >= 8 else 'train'
            if features is None:
                continue
            with open(args.csvdir + '/{}-{}.csv'.format(mode, label), 'a') as f:
                f.write(features.to_csv(index=False, header=False))
            unknown_words += message_unknown_words
    if unknown_words:
        with open(args.csv_dir + '/unknown_words.csv', 'wb') as f:
            unkw_df = pd.DataFrame(
                unknown_words
            ).reset_index().groupby(0).count().sort_values(
                'index', ascending=False)
            f.write(unkw_df.to_csv(header=False).encode('utf8'))
