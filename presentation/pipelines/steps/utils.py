import tensorflow as tf
import logging
import json
import os

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def save_scalar(writer, value, step, name=''):
    with writer.as_default():
        tf.summary.scalar(name, value.result(), step=step)
        
@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_mean(losses)

@tf.function
def custom_acc(y_true, y_pred):
    y_pred  = tf.nn.softmax(y_pred)
    y_true = tf.argmax(y_true, 1, output_type=tf.int32)
    y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
    correct = tf.math.equal(y_true, y_pred)
    correct = tf.cast(correct, tf.float32)
    return tf.reduce_mean(correct)

@tf.function
def train_step(model, batch, opt):
    x, y = batch
    with tf.GradientTape() as tape:
        y_pred = model(x)
        ce = custom_bce(y_true=y, y_pred=y_pred)
        acc = custom_acc(y, y_pred)
    grads = tape.gradient(ce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return acc, ce

@tf.function
def valid_step(model, batch, return_pred=False):
    x, y = batch
    with tf.GradientTape() as tape:
        y_pred = model(x, training=False)
        ce = custom_bce(y_true=y,
                        y_pred=y_pred)
        acc = custom_acc(y, y_pred)
    if return_pred:
        return acc, ce, y_pred, y
    return acc, ce

def train_classifier(model,
          train_batches,
          valid_batches,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          lr=1e-3,
          verbose=1):
    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'validation'))
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(lr)
    # To save metrics
    train_bce  = tf.keras.metrics.Mean(name='train_bce')
    valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # ==============================
    # ======= Training Loop ========
    # ==============================
    best_loss = 999999.
    es_count = 0
    pbar = tqdm(range(epochs), desc='epoch')
    for epoch in pbar:
        for train_batch in train_batches:
            acc, bce = train_step(model, train_batch, optimizer)
            train_acc.update_state(acc)
            train_bce.update_state(bce)

        for valid_batch in valid_batches:
            acc, bce = valid_step(model, valid_batch)
            valid_acc.update_state(acc)
            valid_bce.update_state(bce)

        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_bce, epoch, name='xentropy')
        save_scalar(valid_writter, valid_bce, epoch, name='xentropy')

        if valid_bce.result() < best_loss:
            best_loss = valid_bce.result()
            es_count = 0.
            model.save_weights(os.path.join(exp_path, 'weights'))
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break


        msg = 'EPOCH {} - ES COUNT: {}/{} Train acc: {:.4f} - Val acc: {:.4f} - Train CE: {:.2f} - Val CE: {:.2f}'.format(
                                                                                      epoch,
                                                                                      es_count,
                                                                                      patience,
                                                                                      train_acc.result(),
                                                                                      valid_acc.result(),
                                                                                      train_bce.result(),
                                                                                      valid_bce.result())

        pbar.set_description(msg)

        valid_bce.reset_states()
        train_bce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()
    return model

def get_conf(path):
    conf_file = os.path.join(path, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)
        return conf

def load_weights(model, weigths):
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    return model

def predict_clf(model, test_batches):
    predictions = []
    true_labels = []
    for batch in tqdm(test_batches, desc='test'):
        acc, ce, y_pred, y_true = valid_step(model, batch, return_pred=True)
        if len(y_pred.shape)>2:
            predictions.append(y_pred[:, -1, :])
        else:
            predictions.append(y_pred)

        true_labels.append(y_true)

    y_pred = tf.concat(predictions, 0)
    y_true = tf.concat(true_labels, 0)
    pred_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y_true, 1)
    precision, \
    recall, \
    f1, _ = precision_recall_fscore_support(true_labels,
                                            pred_labels,
                                            average='macro')
    acc = accuracy_score(true_labels, pred_labels)
    results = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'accuracy':acc,
               'y_true':y_true,
               'y_pred':pred_labels}

    return results


def predict_from_path(path, test_batches, mode=0, save=False):
    conf_rnn = get_conf(path)

    if mode == 0:
        clf = get_lstm_attention(conf_rnn['units'],
                               conf_rnn['num_classes'],
                               conf_rnn['w'],
                                     conf_rnn['dropout'])
    if mode == 1:
        clf = get_fc_attention(conf_rnn['units'],
                                     conf_rnn['num_classes'],
                                     conf_rnn['w'])
    if mode == 2:
        clf = get_lstm_no_attention(conf_rnn['units'],
                                    conf_rnn['num_classes'],
                                    conf_rnn['max_obs'],
                                    conf_rnn['dropout'])

    clf = load_weights(clf, path)
    results = predict(clf, test_batches)

    return results