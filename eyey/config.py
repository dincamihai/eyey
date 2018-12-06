import imaplib
from credentials import USER, PASSWORD, SERVER


EVAL_INTERVAL = 60


SUBJECT_FEATURES = 10
BODY_FEATURES = 100


BUGZILLA_HEADERS = [
    "X-Bugzilla-Reason", # QAcontact
    "X-Bugzilla-Type", # changed
    "X-Bugzilla-Watch-Reason", # None
    "X-Bugzilla-Classification", # SUSE Manager
    "X-Bugzilla-Product", # SUSE Manager 3.2
    "X-Bugzilla-Component", # Salt
    "X-Bugzilla-Version", # 3.2.4
    # "X-Bugzilla-Keywords", # DSLA_REQUIRED, DSLA_SOLUTION_PROVIDED
    "X-Bugzilla-Severity", # Major
    "X-Bugzilla-Who", # someone@example.com
    "X-Bugzilla-Status", # REOPENED
    "X-Bugzilla-Priority", # P2 - High
    "X-Bugzilla-Assigned-To", # someone@example.com
    "X-Bugzilla-QA-Contact", # someone@example.com
    "X-Bugzilla-Target-Milestone", # ---
    "X-Bugzilla-Flags", # 
    # "X-Bugzilla-Changed-Fields", # 
    # "X-Bugzilla-NTS-Support-Num", # 
]


def get_connection():
    con = imaplib.IMAP4_SSL(SERVER)
    con.login(USER, PASSWORD)
    return con
