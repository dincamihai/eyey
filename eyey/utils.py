import re
import email
import logging

fh = logging.FileHandler('eyey.log')
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.DEBUG)
log = logging.getLogger('eyeylog')
log.addHandler(fh)
log.addHandler(sh)
log.setLevel(logging.DEBUG)


def get_label(flags):
    if b'should-be-important' in flags:
        return 1
    elif b'should-be-not-important' in flags:
        return 0
    else:
        return None


def get_messages(con, folder, search):
    # typ, data = con.fetch(b'1', "(UID BODY[TEXT])")
    typ, msgnums = con.search(None, search)
    for num in msgnums[0].split():
        result, data = con.fetch(num, "(UID FLAGS RFC822)")# '(UID FLAGS RFC822.HEADER BODY[TEXT])')
        assert result == 'OK'
        info, body = data[0]
        has_flags = re.search(
            b'FLAGS \((?P<flags>.*)\)', info
        )
        flags = has_flags.group('flags').split() if has_flags else []
        raw = email.message_from_bytes(body)
        yield num, raw, flags
