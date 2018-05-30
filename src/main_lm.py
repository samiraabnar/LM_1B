import tensorflow as tf

from lm_1b import LM_1B
from utils import *

import tensorflow as tf

def get_default_hparams():
    return tf.contrib.training.HParams(
        char_vocab_size=256,
        char_embedding_size=16,
        word_embedding_size=1024,
        max_word_length=50,
        batch_size=1,
        sequence_length=1,  # number of timesteps, number of unroll steps, etc
        lstm_num_layers=2,
        vocab_size=800000,
        learning_rate=0.2,
        lstm_clip_grad_norm=1.0,
        chars_padding_id=4,
        chars_bow_id=2,  # begining of word
        chars_eow_id=3,  # end of word
        chars_unknown_id=0,
        tokens_unknown="<UNK>",
        tokens_bos="<S>",  # begining of sentence
        tokens_eos="</S>",  # end of sentence
        tokens_padding="",
        vocab_path="../data/vocab-2016-09-10.txt",

    )

hparams= get_default_hparams()

hparams.sequence_length = 1
hparams.max_word_length = 50
hparams.chars_padding_id = 4

def main(unused_argv):
    lm_model = LM_1B(hparams)
    graph = tf.Graph()
    with graph.as_default():
        lm_model.build_model()



if __name__ == '__main__':
    tf.app.run()