import constants
from utils import get_messages, log, get_label
from config import get_connection

def clean(con):
    folders = [
        'INBOX/train/important',
        'INBOX/test/important',
        'INBOX/train/not-important',
        'INBOX/test/not-important',
    ]
    for folder in folders:
        con.select(folder)
        for (uid, msg, flags) in get_messages(con, folder, 'ALL'):
            result = con.store(uid, "+FLAGS", '\\Deleted')
            log.info('{} {}'.format(uid, result))
    con.expunge()

if __name__ == "__main__":
    folders = ['INBOX']
    unknown_words = []

    con = get_connection()
    clean(con)

    for folder in folders:
        log.info(folder)
        con.select(folder, readonly=True)
        for (uid, msg, flags) in get_messages(con, folder, 'ALL'):
            log.info("%s %s", uid, flags)
            label = get_label(flags)
            if label is None:
                continue
            mode = 'test' if (int(uid) % 10) >= 8 else 'train'

            result = con.copy(uid, 'INBOX/{0}/{1}'.format(mode, 'important' if label else 'not-important'))
