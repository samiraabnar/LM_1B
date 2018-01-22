import tensorflow as tf


class CharCNN(object):
    def __init__(self,hparams):
        self.hparams = hparams


    def build_model(self):
        char_embeddings_lookup = tf.get_variable("W", shape=(self.hparams.char_vocab_size, self.hparams.char_embedding_size),
                                                 dtype=tf.float32,
                                                 initializer=tf.random_uniform_initializer)

        char_embeddings = tf.nn.embedding_lookup(char_embeddings_lookup, char_inputs)
        char_embeddings_reshaped = tf.reshape(char_embeddings,
                                              shape=[-1, self.hparams.max_word_length, self.hparams.char_embedding_size, 1])

        # Parameters found in original lm1b model
        kernels = [1, 2, 3, 4, 5, 6, 7, 7]
        kernel_features = [32, 32, 64, 128, 256, 512, 1024, 2048]

        cnn_layers = []
        for i, (cur_kernel_size, cur_kernel_feature_size) in enumerate(zip(kernels, kernel_features)):
            cnn_cell = self.cnn_cell(char_embeddings_reshaped,
                                        filter_size=cur_kernel_size,
                                        filter_feature_size=cur_kernel_feature_size,
                                        name="charcnn_" + str(i))
            cnn_layers.append(cnn_cell)

        cnn_outputs = tf.concat(cnn_layers, 1)
        self.cnn_outputs = tf.reshape(cnn_outputs, (self.hparams.batch_size * self.hparams.sequence_length, -1))

        self.highway_0 = self.highway_layer(cnn_outputs, num_shards=num_shards, name="dense_0")
        self.highway_1 = slef.highway_layer(self.highway_0, num_shards=num_shards, name="dense_1")

        with tf.variable_scope("proj"):
            proj = sharded_linear(self.highway_1, ((self.highway_1.shape[1].value / num_shards), self.hparams.word_embedding_size),
                                  num_shards=num_shards)

        self.word_embeddings = tf.reshape(proj, (self.hparams.batch_size, self.hparams.sequence_length, self.hparams.word_embedding_size))





    def cnn_cell(self,cnn_inputs, filter_size, filter_feature_size,name="charcnn"):
        with tf.variable_scope(name):
            cnn_filters = tf.get_variable("filter",
                                     shape=(filter_size,self.hparams.char_embedding_size,1,filter_feature_size)
                                     ,dtype=tf.float32)
            cnn_bias = tf.get_variable("cnn_bias", [1,1,1,filter_feature_size])
            conv = tf.nn.conv2d(input=cnn_inputs,
                                filter=cnn_filters
                                ,strides=[1,1,1,1],
                                padding="VALID") + cnn_bias
            charcnn_out = tf.nn.relu(conv)
            charcnn_out = tf.reduce_mac(charcnn_out, reduction_indices=[1],
                                        keep_dims=True)
            charcnn_out = tf.reshape(charcnn_out, shape=[-1, filter_feature_size] )

            return charcnn_out


    def highway_layer(self,input, num_shards= 8, name="dense"):
        with tf.variable_scope(name):
            t_gate = tf.nn.sigmoid(sharded_linear(input,
                                                      (input.shape[1] / num_shards, input.shape[1]), num_shards=num_shards ) - 2.0 )
            tr_gate = tf.nn.relu(
                    sharded_linear(input, (input.shape[1] / num_shards, input.shape[1]), num_shards=num_shards))

            return tf.multiply(t_gate, tr_gate) + tf.multiply((1 - t_gate), input)



