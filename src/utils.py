import tensorflow as tf
import json


def slurp(file_path):
    with open(file_path, 'r') as fin:
        source_lines = map(str.strip, fin.readlines())
        for s in source_lines:
            yield s


def load_config(file_path):
    return json.loads("\n".join(slurp(file_path)))


def create_sharded_weights(shape, num_shards, name="W", concat_dim=0):
    """
    todo: see if tf's built in variable sharding can replace
    :param shape:
    :param num_shards:
    :param name:
    :param concat_dim:
    :return:
    """

    weights = []
    for i in range(0, num_shards):
        cur_w = tf.get_variable(name + "_" + str(i), shape=shape,
                                initializer=tf.random_normal_initializer,
                                dtype=tf.float32)
        weights.append(cur_w)
    w = tf.concat(weights, concat_dim)
    return w


def sharded_linear(input, shape, num_shards):
    """
    todo: see if tf's built in variable sharding can replace
    :param input:
    :param shape:
    :param num_shards:
    :return:
    """
    w = create_sharded_weights(shape, num_shards=num_shards)
    b = tf.get_variable("b", shape=(1, shape[1],), dtype=tf.float32)
    return tf.matmul(input, w) + b


def merge(*dictionaries):
    """
    Helper toward immutable dictionary 'updates' (ala assoc or merge-ing a map in clojure)?
    :param dictionaries: one or more dictionaries to merge, left to right
    :return: single dict
    """
    output = {}
    for d in dictionaries:
        output.update(d)
    return output


def encode_token(token, char_to_id_lookup_table, hparams):
    """
    Encode a word to a padded vector of character IDs
    :param token: tensor of strings representing single word / tokens
    :param char_to_id_lookup_table: Lookup table mapping characters to ids. See utils.vocab for reference
    :param hparams:
    :return: tensor of shape [len(token), max_word_length], representing each word as a vector of character IDs
    """
    max_word_length = hparams.max_word_length
    chars_padding_id = hparams.chars_padding_id
    tokens_bos = hparams.tokens_bos  # begining of sentence
    tokens_eos = hparams.tokens_eos  # end of sentence
    chars_bow_id = hparams.chars_bow_id  # begining of word
    chars_eow_id = hparams.chars_eow_id  # end of word

    s = token
    # Special handling of sentence start and end tokens, per existing model
    s = tf.where(tf.equal(s, tokens_bos), tf.fill(tf.shape(s), chr(0)), s)
    s = tf.where(tf.equal(s, tokens_eos), tf.fill(tf.shape(s), chr(1)), s)
    # Add start and end of word symbols
    s = tf.map_fn(lambda x: tf.constant(chr(chars_bow_id)) + x + tf.constant(chr(chars_eow_id)), s)
    # Add padding
    padding_token = chr(chars_padding_id) * max_word_length
    s = tf.map_fn(lambda x: tf.substr(x + tf.constant(padding_token), 0, max_word_length), s)
    # Split tokens into characters
    s = tf.sparse_tensor_to_dense(tf.string_split(s, delimiter=""), default_value=chr(chars_padding_id))
    # Convert characters to char IDs
    s = char_to_id_lookup_table.lookup(s)
    return s


def encode_sequence(seq, char_to_id_lookup_table, hparams):
    """
    Encode strings to padded character vectors.
    Note: Original model didn't pad sequences, it was trained with one long sequences of sentences concated together.
    Will need to do something slightly different for re-training.
    :param seq: tensor of strings representing a sequence of words (e.g. a sentence). Should typically be wrapped with
    <S> and </S>, but this doesn't expect it.
    :param char_to_id_lookup_table: Lookup table mapping characters to ids. See utils.vocab for reference
    :param hparams:
    :return:
    """

    sequence_length = hparams.sequence_length
    # Split by word
    seq = tf.string_split(seq)
    seq = tf.sparse_tensor_to_dense(seq, default_value=hparams.tokens_padding)
    # Pad to sequence length
    seq_padding = tf.fill([sequence_length], tf.constant(hparams.tokens_padding))
    seq = tf.map_fn(lambda x: tf.concat([x, seq_padding], 0)[:sequence_length], seq)
    # Encode words to character vectors
    # set dtype since this transforms strings to int64s
    seq = tf.map_fn(lambda x: encode_token(x, char_to_id_lookup_table=char_to_id_lookup_table, hparams=hparams), seq,
                    dtype=tf.int64)
    return seq
