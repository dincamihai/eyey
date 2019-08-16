import shutil
import argparse
import datetime
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

from config import EVAL_INTERVAL, BODY_FEATURES, SUBJECT_FEATURES, BUGZILLA_HEADERS
from params import PARAMS

tf.set_random_seed(1234)


CSV_COLUMNS = (
    ['from', 'return', 'to', 'cc', 'x_spam_flag', 'x_spam_score'] +
    ['github_sender', 'github_recipient', 'github_reason'] +
    [it.lower().replace('-', '_') for it in BUGZILLA_HEADERS] +
    ["x_bugzilla_keywords%s" % i for i in range(3)] +
    ["x_bugzilla_changed_fields%s" % i for i in range(10)] +
    ["subject%s" % i for i in range(SUBJECT_FEATURES)] +
    ["body%s" % i for i in range(BODY_FEATURES)] +
    ["timestamp"] +
    ['label']
)


DEFAULTS = (
    ['', '', '', '', '', 0.0] +
    ['', '', ''] +
    ['' for i in range(len(BUGZILLA_HEADERS))] +
    ['', '', ''] +
    [''] * 10 +
    ['' for i in range(SUBJECT_FEATURES)] +
    ['' for i in range(BODY_FEATURES)] +
    [0.0] +
    [0]
)


LABEL_COLUMN = 'label'


def decode_csv(value_column):
    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)
    # No need to features.pop('key') since it is not specified in the INPUT_COLUMNS.
    # The key passes through the graph unused.
    return features, label


def read_dataset(filename, mode, batch_size=512):

      # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(filename)
      # Read lines from text files
      mails_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
      # Parse text lines as comma-separated values (CSV)
      dataset = mails_dataset.map(decode_csv)

      # Note:
      # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
      # use tf.data.Dataset.map to apply one to one transformations (here: text line -> feature list)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = None # indefinitely
          dataset = dataset.shuffle(buffer_size=10 * batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs).batch(batch_size)

      return dataset


def get_feature_columns():
    from_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('from', 1000),
        20
    )
    return_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('return', 1000),
        20
    )
    to_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('to', 1000),
        20
    )
    cc_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('cc', 1000),
        20
    )
    x_spam_flag_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list('x_spam_flag', ['YES', 'NO'])
    )
    x_spam_score_col = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('x_spam_score'), (-50, 0, 50)
    )
    github_sender_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('github_sender', 1000),
        20
    )
    github_recipient_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('github_recipient', 1000),
        20
    )
    github_reason_col = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'github_reason',
            [
                'mention',
                'subscribed',
                'push',
                'assign',
                'author',
                'review_requested',
                'comment',
            ],
        ),
        2
    )

    def make_categorical_column(name, embtype, coltype, colargs, embargs):
        return getattr(tf.feature_column, embtype)(
            getattr(tf.feature_column, coltype)(name, *colargs),
            *embargs
        )

    bugzilla_changed_fields_vocabulary = [
        'status_whiteboard',
        'bug_severity'
        'flagtypes.name',
        'bug_id',
        'short_desc',
        'classification',
        'product',
        'version',
        'rep_platform',
        'op_sys',
        'bug_status',
        'priority',
        'component',
        'assigned_to',
        'reporter',
        'cc'
    ]
    bugzilla_reason = make_categorical_column(
        "x_bugzilla_reason", # qacontact
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [10], [2]
        'categorical_column_with_vocabulary_list',
        [['QAContact', 'CC AssignedTo', 'AssignedTo', 'CC']], [2]
    )
    bugzilla_type = make_categorical_column(
        "x_bugzilla_type", # changed
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [10], [2]
        'categorical_column_with_vocabulary_list',
        [['request', 'whine', 'changed', 'new']], [2]
    )
    bugzilla_component = make_categorical_column(
        "x_bugzilla_component",
        'embedding_column',
        'categorical_column_with_vocabulary_list',
        [[
            'Salt',
            'Other',
            'Maintenance',
            'Server',
            'Containers',
            'UI/UX',
            'Incidents',
            'Client'
        ]], [2]
    )
    bugzilla_who = make_categorical_column(
        "x_bugzilla_who", # someone@example.com
        'embedding_column',
        'categorical_column_with_hash_bucket', [1000], [20]
    )
    bugzilla_status = make_categorical_column(
        "x_bugzilla_status", # reopened
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [10], [2]
        'categorical_column_with_vocabulary_list',
        [['NEW', 'CONFIRMED', 'RESOLVED', 'IN_PROGRESS', 'REOPENED']], [2]
    )
    bugzilla_priority = make_categorical_column(
        "x_bugzilla_priority", # p2 _ high
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [5], [2]
        'categorical_column_with_vocabulary_list',
        [['P1 - Urgent', 'P2 - High', 'P3 - Medium', 'P4 - Low', 'P5 - None']], [2]
    )
    bugzilla_assigned_to = make_categorical_column(
        "x_bugzilla_assigned_to", # someone@example.com
        'embedding_column',
        'categorical_column_with_hash_bucket', [1000], [20]
    )
    bugzilla_keywords0 = make_categorical_column(
        "x_bugzilla_keywords0", # dsla_required, dsla_solution_provi[ded
        "embedding_column",
        'categorical_column_with_vocabulary_list',
        [['DSLA_REQUIRED']], [1]
    )
    bugzilla_keywords1 = make_categorical_column(
        "x_bugzilla_keywords1", # dsla_required, dsla_solution_provi[ded
        "embedding_column",
        'categorical_column_with_vocabulary_list',
        [['DSLA_SOLUTION_PROVIDED']], [1]
    )

    bugzilla_watch_reason = make_categorical_column(
        "x_bugzilla_watch_reason", # none
        'embedding_column', 'categorical_column_with_hash_bucket', [100], [10]
    )

    bugzilla_classification = make_categorical_column(
        "x_bugzilla_classification", # suse manager
        'embedding_column',
        'categorical_column_with_vocabulary_list',
        [[
            'SUSE Linux Enterprise Server',
            'SUSE Manager',
            'openSUSE',
            'Novell Products'
        ]], [2]
    )

    bugzilla_products = make_categorical_column(
        "x_bugzilla_product", # suse manager 3.2
        'embedding_column', 'categorical_column_with_hash_bucket', [100], [10]
    )

    bugzilla_severity = make_categorical_column(
        "x_bugzilla_severity", # major
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [5], [2]
        'categorical_column_with_vocabulary_list',
        [['Major', 'Normal', 'Minor']], [2]
    )
    bugzilla_qa_contact = make_categorical_column(
        "x_bugzilla_qa_contact", # someone@example.com
        'embedding_column', 'categorical_column_with_hash_bucket', [1000], [20]
    )
    bugzilla_flags = make_categorical_column(
        "x_bugzilla_flags", # 
        'embedding_column',
        'categorical_column_with_vocabulary_list',
        [['needinfo?', 'needinfo? needinfo?', 'needinfo? needinfo? needinfo?']], [2]
    )
    bugzilla_changed_fields0 = make_categorical_column(
        "x_bugzilla_changed_fields0", # 
        'embedding_column',
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields1 = make_categorical_column(
        "x_bugzilla_changed_fields1", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields2 = make_categorical_column(
        "x_bugzilla_changed_fields2", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields3 = make_categorical_column(
        "x_bugzilla_changed_fields3", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields4 = make_categorical_column(
        "x_bugzilla_changed_fields4", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields5 = make_categorical_column(
        "x_bugzilla_changed_fields5", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields6 = make_categorical_column(
        "x_bugzilla_changed_fields6", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields7 = make_categorical_column(
        "x_bugzilla_changed_fields7", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields8 = make_categorical_column(
        "x_bugzilla_changed_fields8", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )
    bugzilla_changed_fields9 = make_categorical_column(
        "x_bugzilla_changed_fields9", # 
        'embedding_column',
        # 'categorical_column_with_hash_bucket', [1000], [20]
        'categorical_column_with_vocabulary_list',
        [bugzilla_changed_fields_vocabulary], [3]
    )

    subject_cols = [
        # tf.feature_column.indicator_column(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('subject%s' % i, 10000),
            50
        ) for i in range(SUBJECT_FEATURES)
    ]
    body_cols = [
        # tf.feature_column.indicator_column(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('body%s' % i, 10000),
            50
        ) for i in range(BODY_FEATURES)
    ]
    source_target_crossed_col = tf.feature_column.embedding_column(
        tf.feature_column.crossed_column(
            ['from', 'to', 'cc'], 1000
        ),
        50
    )
    body_crossed_col = tf.feature_column.embedding_column(
        tf.feature_column.crossed_column(
            # ['subject%s' %i for i in range(SUBJECT_FEATURES)] +
            ['body%s' %i for i in range(BODY_FEATURES)], 10000
        ),
        50
    )
    now = datetime.datetime.now()
    one_year_ago = (now - datetime.timedelta(365)).timestamp()
    one_month_ago = (now - datetime.timedelta(30)).timestamp()
    timestamp_col = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('timestamp'),
            (one_year_ago, one_month_ago, now.timestamp())
    ),
        20
    )
    return (
        [
            # timestamp_col,
            from_col,
            to_col,
            cc_col,
            source_target_crossed_col,
            x_spam_flag_col,
            # x_spam_score_col,
            github_recipient_col,
            github_reason_col,
            bugzilla_reason,
            # bugzilla_type,
            bugzilla_component,
            # bugzilla_who,
            bugzilla_status,
            # bugzilla_priority,
            bugzilla_assigned_to,
            # bugzilla_keywords0,
            # bugzilla_keywords1,
            body_crossed_col,
        ] +
        # subject_cols +
        body_cols
    )


def read_csv_row_in():
    csv_row = tf.placeholder(shape=[None], dtype=tf.string, name='csv_row')
    features, _ = decode_csv(csv_row)
    return features, csv_row


def serving_input_receiver_fn():
    features, csv_row = read_csv_row_in()
    return tf.estimator.export.ServingInputReceiver(
        features, {'csv_row': csv_row})


NCLASSES=2


def apply_batch_norm(layer, mode, params):
    if params.get('batch_norm', False):
        layer = tf.layers.batch_normalization(
            layer, training=(mode == tf.estimator.ModeKeys.TRAIN)) #only batchnorm when training
    layer = params.get('activation', tf.nn.relu)(layer)
    return layer


def model_fn(features, labels, mode, params):

    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    #apply batch normalization
    # l1 = tf.layers.dense(input_layer, 100, activation=None)
    # l1 = apply_batch_norm(l1, mode, params)
    # l2 = tf.layers.dense(l1, 300, activation=None)
    # l2 = apply_batch_norm(l2, mode, params)
    # l3 = tf.layers.dense(l2, 100, activation=None)
    # l3 = apply_batch_norm(l3, mode, params)
    # l4 = tf.layers.dense(l3, 10, activation=None)
    # l4 = apply_batch_norm(l4, mode, params)
    ylogits = tf.layers.dense(input_layer, NCLASSES, activation=None)

    predictions = tf.math.argmax(ylogits, 1)

    loss = None
    train_op = None
    evalmetrics = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, ylogits)
        # optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        optimizer = tf.train.ProximalAdagradOptimizer(
            l1_regularization_strength=params['l1_regularization'],
            l2_regularization_strength=params['l2_regularization'],
            learning_rate=params['learning_rate']
        )
        optimizer = near_optimizer=tf.train.FtrlOptimizer(
            l1_regularization_strength=params['l1_regularization'],
            l2_regularization_strength=params['l2_regularization'],
            learning_rate=params['learning_rate']
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        def mcc(labels, predictions):
            TP = tf.count_nonzero(predictions * labels)
            TN = tf.count_nonzero((predictions - 1) * (labels - 1))
            FP = tf.count_nonzero(predictions * (labels - 1))
            FN = tf.count_nonzero((predictions - 1) * labels)
            ret = (TP * TN - FP * FN) / tf.math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            return ret

        evalmetrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions),
            'precision': tf.metrics.precision(labels, predictions),
            'recall': tf.metrics.recall(labels, predictions),
            'f1_score': tf.contrib.metrics.f1_score(labels, predictions),
            # 'mcc': mcc(labels, predictions)
        }


    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predictions},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=evalmetrics,
        export_outputs={"predictions": tf.estimator.export.PredictOutput(predictions)}
    )


def get_estimator(output_dir, hparams, learning_rate=None):
    linear_optimizer = near_optimizer=tf.train.FtrlOptimizer(
        l1_regularization_strength=0.06,
        l2_regularization_strength=0.06,
        learning_rate=learning_rate or 0.1
    )
    dnn_optimizer = tf.train.ProximalAdagradOptimizer(
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.03,
        learning_rate=learning_rate or 0.1
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL),
        model_dir=output_dir)

    # estimator = tf.estimator.LinearClassifier(
    #     feature_columns=feature_cols,
    #     # optimizer=linear_optimizer,
    #     # hidden_units=[120, 120, 120],
    #     # linear_feature_columns=feature_cols[:4] + feature_cols[-4:],
    #     # linear_optimizer=linear_optimizer,
    #     # dnn_feature_columns=feature_cols[4:-4],
    #     # dnn_hidden_units=[150, 100, 100],
    #     #dnn_optimizer=dnn_optimizer,
    #     optimizer=linear_optimizer,
    #     model_dir=output_dir)

    # def get_metrics(labels, predictions):
    #     predictions = tf.expand_dims(tf.cast(predictions['predictions'], tf.float64), -1)
    #     return {
    #         'f1_score': tf.contrib.metrics.f1_score(labels=labels, predictions=predictions),
    #     }

    # estimator = tf.contrib.estimator.add_metrics(estimator, get_metrics)

    return estimator


def train_and_evaluate(output_dir, hparams, csv_dir="./data"):
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = get_estimator(output_dir, hparams, learning_rate=0.06)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: read_dataset(csv_dir + '/train-*.csv', tf.estimator.ModeKeys.TRAIN, batch_size=hparams['batch_size']),
        max_steps=hparams['num_train_steps'])

    exporter = tf.estimator.LatestExporter('exporter', serving_input_receiver_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: read_dataset(csv_dir + '/test-*.csv', tf.estimator.ModeKeys.EVAL, batch_size=hparams['batch_size']),
        steps=None,
        start_delay_secs=1,  # start evaluating after N seconds
        throttle_secs=5,  # evaluate every N seconds
        exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def post_train_and_evaluate(features, output_dir, csv_dir):
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = get_estimator(output_dir, 0.1)

    def fn(df):
        types = {}
        types.update({k: str for k in ['to', 'from', 'cc', 'return']})
        types.update({"subject%s" % i: str for i in range(SUBJECT_FEATURES)})
        types.update({"body%s" % i: str for i in range(BODY_FEATURES)})
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(df.iloc[:, :-1].astype(types)),
            df['label'].tolist())
        )
        return dataset.repeat(1).batch(128)

    estimator.train(lambda: fn(features), steps=1)

    estimator.evaluate(
        input_fn=lambda: read_dataset(
            csv_dir + '/test-*.csv', tf.estimator.ModeKeys.EVAL, batch_size=32),
        steps=None)

    estimator.export_saved_model(
        output_dir + '/export/exporter',
        serving_input_receiver_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None)

    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: fn(features), max_steps=None)

    # exporter = tf.estimator.LatestExporter(
    #     'exporter', serving_input_receiver_fn)

    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: read_dataset(
    #         'data/test-*.csv', tf.estimator.ModeKeys.EVAL, batch_size=128),
    #     steps=None,
    #     start_delay_secs=1, # start evaluating after N seconds
    #     throttle_secs=10,  # evaluate every N seconds
    #     exporters=exporter)

    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='default')
    args = parser.parse_args()
    shutil.rmtree(PARAMS[args.params]['outdir'], ignore_errors=True) # start fresh each time
    train_and_evaluate(
        PARAMS[args.params]['outdir'],
        hparams={
            'num_train_steps': 5200,
            'feature_columns': get_feature_columns(),
            'learning_rate': 0.01,
            'l1_regularization': 0.0,
            'l2_regularization': 0.3,
            # 'batch_norm': True,
            'activation': tf.nn.relu,
            'batch_size': 16,
        },
        csv_dir=PARAMS[args.params]['csvdir']
    )
