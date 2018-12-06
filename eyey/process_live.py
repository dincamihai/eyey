import re
import datetime
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from utils import get_messages, log
from config import BODY_FEATURES, SUBJECT_FEATURES, BUGZILLA_HEADERS


def clean_content(content):
    try:
        return content.rstrip(string.punctuation).lower()
    except Exception as ex:
        log.warning(ex)


def get_content(message):
    if message.is_multipart():
        parts = message.get_payload()
        try:
            content = b''.join([(part.get_payload(decode=True) or b'') for part in parts])
        except Exception as ex:
            log.warning(ex)
    else:
        content = message.get_payload(decode=True)
    content = content.decode('utf8', 'ignore')
    return clean_content(content)


def get_words(tokens, porter):
    words = []
    unknown_words = []
    for token in tokens:
        if token.isalpha() and token not in stopwords.words('english'):
            try:
                words.append((porter.stem(token), token))
            except KeyError as kerr:
                unknown_words.append(token)
                continue
    return words, unknown_words


def get_features(words, limit):
    # columns = ['stem'] + [i for i in range(50)]
    columns = ['stem']
    words_df = pd.DataFrame(
        [[row[0]] for row in words], columns=columns)
    top_words = words_df['stem'].tolist()[:limit]
    empty_words = [''] * (limit - len(top_words))
    ret = top_words + empty_words
    assert len(ret) == limit
    return ret


def process_content(content, porter, limit):
    is_html = bool(BeautifulSoup(content, "html.parser").find())
    if is_html:
        soup = BeautifulSoup(content,'lxml')
        content = soup.get_text()
    try:
        tokens = word_tokenize(content)
    except Exception as exc:
        log.warning(exc)
        return None, []

    words, unknown_words = get_words(tokens, porter)

    return get_features(words, limit), unknown_words


def _extract_mail(text):
    if not text:
        return ''
    mail_regex = re.compile(
        r'<*(?P<email>[a-zA-Z0-9-=+._]+@[a-zA-Z0-9-_]+\.[a-zA-Z]+)>*')
    res = mail_regex.search(text)
    return res.groupdict()['email'] if res else ''


def process_message(message, porter, label):
    try:
        content = get_content(message)
    except Exception as exc:
        log.warning(exc)
        features, unknown_words = (None, [])
    else:
        content_features, unknown_words = process_content(content, porter, BODY_FEATURES)
        from_ = _extract_mail(message['From'])
        return_ = _extract_mail(message['Return-Path'])
        to_ = _extract_mail(message['To'])
        cc_ = _extract_mail(message['Cc'])
        x_spam_flag = message['X-Spam-Flag']
        x_spam_score = message['X-Spam-Score']
        github_sender = message.get('X-GitHub-Sender', '')
        github_recipient = message.get('X-GitHub-Recipient', '')
        github_reason = message.get('X-GitHub-Reason', '')
        bugzilla_headers_features = []
        for it in BUGZILLA_HEADERS:
            bugzilla_headers_features.append(message.get(it, ''))
        NUM_KEYWORDS = 3
        bugzilla_keywords_features = [it for it in message.get("X-Bugzilla-Keywords", '').split(',')[:NUM_KEYWORDS] if it]
        for i in range(NUM_KEYWORDS - len(bugzilla_keywords_features)):
            bugzilla_keywords_features.append('')
        NUM_CHANGED_FIELDS = 10
        bugzilla_changed_fields_features = [it for it in message.get("X-Bugzilla-Changed-Fields", '').split()[:NUM_CHANGED_FIELDS] if it]
        for i in range(NUM_CHANGED_FIELDS - len(bugzilla_changed_fields_features)):
            bugzilla_changed_fields_features.append('')
        if bugzilla_changed_fields_features[0]:
            log.info(bugzilla_changed_fields_features)
        subject_features, _ = process_content(message['subject'], porter, SUBJECT_FEATURES)
        try:
            timestamp = datetime.datetime.strptime(message['Date'], '%a, %d %b %Y %H:%M:%S %z').timestamp()
        except Exception:
            timestamp = 0.0
        features = pd.DataFrame(
            [
                [from_, to_, cc_, return_, x_spam_flag, x_spam_score] +
                [github_sender, github_recipient, github_reason] +
                bugzilla_headers_features +
                bugzilla_keywords_features +
                bugzilla_changed_fields_features +
                subject_features +
                content_features +
                [timestamp] +
                [label if label else None]
            ],
            columns=(
                ['from', 'return', 'to', 'cc', 'x_spam_flag', 'x_spam_score'] +
                ['github_sender', 'github_recipient', 'github_reason'] +
                [it.lower().replace('-', '_') for it in BUGZILLA_HEADERS] +
                ["x_bugzilla_keywords%s" % i for i in range(NUM_KEYWORDS)] +
                ["x_bugzilla_changed_fields%s" % i for i in range(NUM_CHANGED_FIELDS)] +
                ["subject%s" % i for i in range(len(subject_features))] +
                ["body%s" % i for i in range(len(content_features))] +
                ["timestamp"] +
                ['label']
            )
        )
    return (features, unknown_words)
