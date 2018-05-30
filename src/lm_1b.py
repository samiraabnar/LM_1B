import tensorflow as tf
from tensorflow.python.ops import lookup_ops

import os
import re


from CharCNN import CharCNN
from utils import *

CHAR_EMBEDDING_SCOPE= "char_embedding"
LSTM_SCOPE_PREFIX="lstm/lstm_"
SOFTMAX_SCOPE= "softmax"
NUM_SHARDS=8


class LM_1B(object):

    def __init__(self,hparams):
        self.hparams = hparams




    def build_model(self):
        graph_nodes = {}
        # Attach word / character lookup tables
        self.lookup_nodes = self.vocab_lookup_graph()
        self.char_to_id_lookup_table = self.lookup_nodes['lookup_char_to_id']
        self.word_to_id_lookup_table = self.lookup_nodes['lookup_word_to_id']

        # placeholder for input sentences (encoded as character arrays)
        input_seqs = tf.placeholder(dtype=tf.int64, shape=(self.hparams.sequence_length, self.hparams.max_word_length))
        # attach the model itself
        self.inference_nodes = self.inference(input_seqs)
        # attach a helper to lookup top k predictions
        self.prediction_nodes = self.prediction(self.inference_nodes['logits'],self.lookup_nodes['lookup_id_to_word'],k=10)


    def lstm_cell(self,input):
        cell = tf.contrib.rnn.LSTMCell(num_units=NUM_SHARDS * self.hparams.word_embedding_size,
                                       num_proj=self.hparams.word_embedding_size,
                                       num_unit_shards=NUM_SHARDS, num_proj_shards=NUM_SHARDS,
                                       forget_bias=1.0, use_peepholes=True)

        state_c = tf.get_variable(name="state_c",
                                  shape=(self.hparams.batch_size * self.hparams.sequence_length, 8192),
                                  initializer=tf.zeros_initializer,
                                  trainable=False)
        state_h = tf.get_variable(name="state_h",
                                  shape=(self.hparams.batch_size * self.hparams.sequence_length, 1024),
                                  initializer=tf.zeros_initializer,
                                  trainable=False)

        out_0, state_0 = cell(input, tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h))

        ass_c = tf.assign(state_c, state_0[0])
        ass_h = tf.assign(state_h, state_0[1])

        with tf.control_dependencies([ass_c, ass_h]):
            out_0 = tf.identity(out_0)

        return out_0, state_0


    def projection(self,input):
        softmax_w = create_sharded_weights((self.hparams.vocab_size / NUM_SHARDS, self.hparams.word_embedding_size),
                                           num_shards=NUM_SHARDS,
                                           concat_dim=1)
        softmax_w = tf.reshape(softmax_w, shape=(-1, self.hparams.word_embedding_size))
        softmax_b = tf.get_variable('projection_b', shape=(self.hparams.vocab_size))

        logits = tf.nn.bias_add(tf.matmul(input, softmax_w, transpose_b=True), softmax_b, data_format="NHWC")
        return logits


    def vocab_lookup_graph(self):
        lookup_id_to_word = lookup_ops.index_to_string_table_from_file(self.hparams.vocab_path, default_value=self.hparams.tokens_unknown)
        lookup_word_to_id = lookup_ops.index_table_from_file(self.hparams.vocab_path, default_value=-1)
        all_chars = list(map(lambda i: chr(i), range(0, 255)))
        lookup_char_to_id = lookup_ops.index_table_from_tensor(tf.constant(all_chars),
                                                               default_value=self.hparams.chars_unknown_id)
        lookup_id_to_char = lookup_ops.index_to_string_table_from_tensor(tf.constant(all_chars),
                                                                         default_value=chr(self.hparams.chars_unknown_id))
        return {"lookup_id_to_word": lookup_id_to_word,
                "lookup_word_to_id": lookup_word_to_id,
                "lookup_char_to_id": lookup_char_to_id,
                "lookup_id_to_char": lookup_id_to_char}


    def perplexity(self,targets,target_weights,logits):
        target_list = tf.reshape(targets, [-1])
        target_weights_list = tf.to_float(tf.reshape(target_weights, [-1]))

        # hrmm
        word_count = tf.add(tf.reduce_sum(target_weights_list), 0.0000999999974738)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=target_list)
        cross_entropy = tf.multiply(cross_entropy, tf.to_float(target_weights))

        return {"log_perplexity": tf.reduce_sum(cross_entropy) / word_count,
                "cross_entropy": cross_entropy}



    def train_graph(self):
        trainable_vars = tf.trainable_variables()
        tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="")
        tf.global_variables()

        all_gradients = tf.gradients(self.loss, trainable_vars)

        lstm_gradients = filter(lambda x: -1 < x.op.name.find("lstm"), all_gradients)
        non_lstm_gradients = set(all_gradients).difference(lstm_gradients)

        lstm_gradients, global_norm = tf.clip_by_global_norm(lstm_gradients, self.hparams.lstm_clip_grad_norm)
        all_gradients = non_lstm_gradients.union(lstm_gradients)

        optimizer = tf.train.AdagradOptimizer(self.hparams.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = optimizer.apply_gradients(zip(all_gradients, trainable_vars), global_step=global_step)

        return {"train_op": train_op, "global_step": global_step}



    def prediction(self,logits,id_to_word_lookup_table,k):
        top_k = tf.nn.top_k(logits, k)
        top_word_ids = top_k.indices
        word_predictions = tf.reshape(id_to_word_lookup_table.lookup(
                                                                tf.to_int64(tf.reshape(top_word_ids, [-1]))),
                                      [-1, k])
        return {"predicted_words": word_predictions,
                "top_k": top_k}


    def inference(self,input_seqs):
        charcnn = CharCNN(self.hparams)
        charcnn.build_model(input_seqs)
        with tf.variable_scope(CHAR_EMBEDDING_SCOPE):
            word_embeddings = charcnn.word_embeddings
            word_embeddings = tf.reshape(word_embeddings, (-1, self.hparams.word_embedding_size))

        cell_out = word_embeddings
        for layer_num in range(0, 2):
            with tf.variable_scope(LSTM_SCOPE_PREFIX + str(layer_num)):
                cell_out, cell_state = self.lstm_cell(cell_out)

        lstm_outputs = tf.reshape(cell_out, shape=(-1, self.hparams.word_embedding_size))

        with tf.variable_scope(SOFTMAX_SCOPE):
            logits = self.projection(lstm_outputs)

        return {
            "word_embeddings": word_embeddings,
            "lstm_outputs": lstm_outputs,
            "lstm_state": cell_state,
            "logits": logits
        }

    def create_lm1b_restoration_var_map(self,char_embedding_vars, lstm_vars, softmax_vars):
        var_map = {}

        # Map char embedding vars
        var_map = merge(var_map, dict(map(lambda x: (x.op.name, x), char_embedding_vars)))

        # Map lstm embedding vars
        var_map_regexes = {r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/projection/kernel/part_(\d).*": r"\1/W_P_\2",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/kernel/part_(\d).*": r"\1/W_\2",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/bias.*": r"\1/B",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/w_([fio])_diag.*":
                               lambda match: match.group(1) + "/W_" + match.group(
                                   2).upper() + "_diag",
                           }
        for r_match, r_replace in var_map_regexes.items():
            matching_variables = filter(lambda x: re.match(r_match, x.name), lstm_vars)
            for v in matching_variables:
                var_map[re.sub(r_match, r_replace, v.name)] = v

        # Map softmax embedding vars
        var_map = merge(var_map, dict(map(lambda x: (x.op.name, x), softmax_vars)))

        return var_map

    def restore_original_lm1b(self,sess, run_config):
        """
        Var mapping shenanigans to restore the pre-trained model to the current graph
        :param sess:
        :param run_config:
        :return:
        """

        var_map = self.create_lm1b_restoration_var_map(
            char_embedding_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=CHAR_EMBEDDING_SCOPE),
            lstm_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=LSTM_SCOPE_PREFIX),
            softmax_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=SOFTMAX_SCOPE)
        )

        saver = tf.train.Saver(var_list=var_map)
        saver.restore(sess, os.path.join(run_config['model_dir_path_original'], "ckpt-*"))