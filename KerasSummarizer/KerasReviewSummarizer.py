import tensorflow as tf
import time
from tensorflow.python.layers.core import Dense
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from datetime import datetime
from .DataPreprocessor import DataPreprocessor


class KerasReviewSummarizer:
    in_verbose_mode = False
    do_print_verbose_header = True

    TEXT_CODES = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

    word_embedding_matrix = None
    epochs = 100
    batch_size = 64
    rnn_size = 256
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.75

    learning_rate_decay = 0.95
    min_learning_rate = 0.0005

    def __init__(self, word_embedding_matrix, in_verbose_mode=False):
        self.word_embedding_matrix = word_embedding_matrix
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")

    def get_model_inputs(self):
        '''Create palceholders for inputs to the model'''
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        keep_probability = tf.placeholder(tf.float32, name='keep_prob')
        summary_length = tf.placeholder(
            tf.int32,
            (None,),
            name='summary_length'
        )
        max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
        text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        model_inputs = {
            "input_data": input_data,
            "targets": targets,
            "learning_rate": learning_rate,
            "keep_probability": keep_probability,
            "summary_length": summary_length,
            "max_summary_length": max_summary_length,
            "text_length": text_length
        }
        return model_inputs

    def process_encoding_input(self, target_data, vocab_to_int, batch_size):
        '''
        Remove the last word id from each batch and
        concat the <GO> to the begining of each batch
        '''
        ending = tf.strided_slice(
            target_data,
            [0, 0],
            [batch_size, -1],
            [1, 1]
        )
        dec_input = tf.concat(
            [tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending],
            1
        )
        return dec_input

    def encoding_layer(
        self,
        rnn_size,
        sequence_length,
        num_layers,
        rnn_inputs,
        keep_prob
    ):
        '''Create the encoding layer'''
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1,
                        0.1,
                        seed=2
                    )
                )
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell_fw,
                    input_keep_prob=keep_prob
                )
                cell_bw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1,
                        0.1,
                        seed=2
                    )
                )
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw,
                    input_keep_prob=keep_prob
                )
                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    rnn_inputs,
                    sequence_length,
                    dtype=tf.float32
                )
        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output, 2)
        return enc_output, enc_state

    def training_decoding_layer(
        self,
        dec_embed_input,
        summary_length,
        dec_cell,
        initial_state,
        output_layer,
        vocab_size,
        max_summary_length
    ):
        '''Create the training logits'''
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=dec_embed_input,
            sequence_length=summary_length,
            time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            training_helper,
            initial_state,
            output_layer
        )
        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_summary_length
        )
        return training_logits

    def inference_decoding_layer(
        self,
        embeddings,
        start_token,
        end_token,
        dec_cell,
        initial_state,
        output_layer,
        max_summary_length,
        batch_size
    ):
        '''Create the inference logits'''
        start_tokens = tf.tile(tf.constant(
            [start_token],
            dtype=tf.int32),
            [batch_size],
            name='start_tokens'
        )
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embeddings,
            start_tokens,
            end_token
        )
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            inference_helper,
            initial_state,
            output_layer
        )
        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_summary_length
        )
        return inference_logits

    def decoding_layer(
        self,
        dec_embed_input,
        embeddings,
        enc_output,
        enc_state,
        vocab_size,
        text_length,
        summary_length,
        max_summary_length,
        rnn_size,
        vocab_to_int,
        keep_prob,
        batch_size,
        num_layers
    ):
        '''Create the decoding cell and attention for
        the training and inference decoding layers'''
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1,
                        0.1,
                        seed=2
                    )
                )
                dec_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm,
                    input_keep_prob=keep_prob
                )
        output_layer = Dense(
            vocab_size,
            kernel_initializer=tf.truncated_normal_initializer(
                mean=0.0, stddev=0.1
            )
        )
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            rnn_size,
            enc_output,
            text_length,
            normalize=False,
            name='BahdanauAttention'
        )
        dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
            dec_cell,
            attn_mech,
            rnn_size
        )
        initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
            enc_state[0],
            _zero_state_tensors(
                rnn_size,
                batch_size,
                tf.float32
            )
        )
        with tf.variable_scope("decode"):
            training_logits = self.training_decoding_layer(
                dec_embed_input,
                summary_length,
                dec_cell,
                initial_state,
                output_layer,
                vocab_size,
                max_summary_length
            )
        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.inference_decoding_layer(
                embeddings,
                vocab_to_int['<GO>'],
                vocab_to_int['<EOS>'],
                dec_cell,
                initial_state,
                output_layer,
                max_summary_length,
                batch_size
            )
        return training_logits, inference_logits

    def seq2seq_model(
        self,
        input_data,
        target_data,
        keep_prob,
        text_length,
        summary_length,
        max_summary_length,
        vocab_size,
        rnn_size,
        num_layers,
        vocab_to_int,
        batch_size
    ):
        '''Use the previous functions to create
        the training and inference logits'''

        # Use Numberbatch's embeddings and
        # the newly created ones as our embeddings
        embeddings = self.word_embedding_matrix

        enc_embed_input = tf.nn.embedding_lookup(
            embeddings,
            input_data
        )
        enc_output, enc_state = self.encoding_layer(
            rnn_size,
            text_length,
            num_layers,
            enc_embed_input,
            keep_prob
        )
        dec_input = self.process_encoding_input(
            target_data,
            vocab_to_int,
            batch_size
        )
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

        training_logits, inference_logits = self.decoding_layer(
            dec_embed_input,
            embeddings,
            enc_output,
            enc_state,
            vocab_size,
            text_length,
            summary_length,
            max_summary_length,
            rnn_size,
            vocab_to_int,
            keep_prob,
            batch_size,
            num_layers
        )
        return training_logits, inference_logits

    def pad_sentence_batch(self, sentence_batch, vocab_to_int):
        """Pad sentences with <PAD> so that
        each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        result = [
            sentence + [vocab_to_int['<PAD>']] * (
                max_sentence - len(sentence)
            )
            for sentence in sentence_batch
        ]
        return result

    def get_batches(self, summaries, texts, vocab_to_int, batch_size):
        """Batch summaries, texts, and
        the lengths of their sentences together"""
        for batch_i in range(0, len(texts)//batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(
                self.pad_sentence_batch(summaries_batch, vocab_to_int)
            )
            pad_texts_batch = np.array(
                self.pad_sentence_batch(texts_batch, vocab_to_int)
            )

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))

            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))

            yield pad_summaries_batch, \
                pad_texts_batch, \
                pad_summaries_lengths, \
                pad_texts_lengths

    def train(self, sorted_texts, sorted_summaries, vocab_to_int):
        # Build the graph
        train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with train_graph.as_default():

            # Load the model inputs
            model = self.get_model_inputs()

            print(tf.reverse(model["input_data"], [-1]))

            return
            # Create the training and inference logits
            training_logits, inference_logits = self.seq2seq_model(
                tf.reverse(model["input_data"], [-1]),
                model["targets"],
                model["keep_probability"],
                model["text_length"],
                model["summary_length"],
                model["max_summary_length"],
                len(vocab_to_int) + 1,
                self.rnn_size,
                self.num_layers,
                vocab_to_int,
                self.batch_size
            )

            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(
                training_logits.rnn_output,
                'logits'
            )
            inference_logits = tf.identity(
                inference_logits.sample_id,
                name='predictions'
            )

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(
                model["summary_length"],
                model["max_summary_length"],
                dtype=tf.float32,
                name='masks'
            )

            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    model["targets"],
                    masks
                )

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var)
                    for grad, var in gradients if grad is not None
                ]
                train_op = optimizer.apply_gradients(capped_gradients)
        print("Graph is built.")

        # ## Training the Model

        # Since I am training this model on my MacBook Pro,
        # it would take me days if I used the whole dataset.
        # For this reason, I am only going to use a subset of the data,
        # so that I can train it over night.
        # Normally I use [FloydHub's](https://www.floydhub.com/)
        # services for my GPU needs, but it would take quite a bit of time to
        # upload the dataset and ConceptNet Numberbatch,
        # so I'm not going to bother with that for this project.
        #
        # I chose not use use the start of the subset because
        # I didn't want to make it too easy for my model.
        # The texts that I am using are closer to the median lengths;
        # I thought this would be more fair.

        # In[234]:

        # Subset the data for training
        start = 200000
        end = start + 50000
        sorted_summaries_short = sorted_summaries[start:end]
        sorted_texts_short = sorted_texts[start:end]
        print("The shortest text length:", len(sorted_texts_short[0]))
        print("The longest text length:", len(sorted_texts_short[-1]))

        # Train the Model
        display_step = 20  # Check training loss after every 20 batches
        stop_early = 0
        stop = 3  # If update loss doesn't decrease in 3 update checks, stop
        per_epoch = 3  # Make 3 update checks per epoch
        update_check = (len(sorted_texts_short)//self.batch_size//per_epoch)-1

        update_loss = 0
        batch_loss = 0
        summary_update_loss = []  # Record updates for saving improvements

        checkpoint = "best_model.ckpt"
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If we want to continue training a previous session
            # loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            # loader.restore(sess, checkpoint)
            for epoch_i in range(1, self.epochs + 1):
                update_loss = 0
                batch_loss = 0
                for batch_i, (
                    summaries_batch,
                    texts_batch,
                    summaries_lengths,
                    texts_lengths
                ) in enumerate(
                    self.get_batches(
                        sorted_summaries_short,
                        sorted_texts_short,
                        vocab_to_int,
                        self.batch_size
                    )
                ):
                    start_time = time.time()
                    _, loss = sess.run(
                        [train_op, cost],
                        {
                            model["input_data"]: texts_batch,
                            model["targets"]: summaries_batch,
                            model["learning_rate"]: self.learning_rate,
                            model["summary_length"]: summaries_lengths,
                            model["text_length"]: texts_lengths,
                            model["keep_probability"]: self.keep_probability
                        }
                    )

                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and batch_i > 0:
                        print(
                            '''Epoch {:>3}/{} Batch {:>4}/{}
                            - Loss: {:>6.3f}, Seconds: {:>4.2f}'''.format(
                                epoch_i,
                                self.epochs,
                                batch_i,
                                len(sorted_texts_short) // self.batch_size,
                                batch_loss / display_step,
                                batch_time * display_step
                            )
                        )
                        batch_loss = 0

                    if batch_i % update_check == 0 and batch_i > 0:
                        print("Average loss for this update:", round(
                            update_loss/update_check, 3)
                        )
                        summary_update_loss.append(update_loss)

                        # If the update loss is at a new minimum,
                        # save the model
                        if update_loss <= min(summary_update_loss):
                            print('New Record!')
                            stop_early = 0
                            saver = tf.train.Saver()
                            saver.save(sess, checkpoint)

                        else:
                            print("No Improvement.")
                            stop_early += 1
                            if stop_early == stop:
                                break
                        update_loss = 0

                # Reduce learning rate, but not below its minimum value
                self.learning_rate *= self.learning_rate_decay
                if self.learning_rate < self.min_learning_rate:
                    self.learning_rate = self.min_learning_rate

                if stop_early == stop:
                    print("Stopping Training.")
                    break

    # ## Making Our Own Summaries

    # To see the quality of the summaries that this model can generate,
    # you can either create your own review, or use a review from the dataset.
    # You can set the length of the summary to a fixed value,
    # or use a random value like I have here.

    def text_to_seq(self, text, vocab_to_int, int_to_vocab):
        '''Prepare the text for the model'''
        data_preprocessor = DataPreprocessor(self.in_verbose_mode)
        text = data_preprocessor.clean_text(text)
        result = [
            vocab_to_int.get(word, int_to_vocab['<UNK>'])
            for word in text.split()
        ]
        return result

    def run(self, clean_texts, vocab_to_int, int_to_vocab):
        # Create your own review or use one from the dataset
        # input_sentence = "I have never eaten an apple before,
        # but this red one was nice.
        # I think that I will try a green apple next time."
        # text = text_to_seq(input_sentence)
        random = np.random.randint(0, len(clean_texts))
        input_sentence = clean_texts[random]
        text = self.text_to_seq(clean_texts[random], vocab_to_int, int_to_vocab)

        checkpoint = "./best_model.ckpt"

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            text_length = loaded_graph.get_tensor_by_name('text_length:0')
            summary_length = loaded_graph.get_tensor_by_name(
                'summary_length:0'
            )
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

            # Multiply by batch_size to match the model's input parameters
            answer_logits = sess.run(
                logits,
                {
                    input_data: [text] * self.batch_size,
                    summary_length: [np.random.randint(5, 8)],
                    text_length: [len(text)] * self.batch_size,
                    keep_prob: 1.0
                }
            )[0]

        pad = vocab_to_int["<PAD>"]

        print('Original Text:', input_sentence)

        print('\nText')
        print('  Word Ids:    {}'.format([i for i in text]))
        print('  Input Words: {}'.format(
            " ".join([int_to_vocab[i] for i in text]))
        )

        print('\nSummary')
        print('  Word Ids:       {}'.format(
            [i for i in answer_logits if i != pad])
        )
        print('  Response Words: {}'.format(
            " ".join([int_to_vocab[i] for i in answer_logits if i != pad]))
        )

    def say(self, message, end="\n"):
        if self.in_verbose_mode is True:
            if self.do_print_verbose_header is True:
                current_time = datetime.now().strftime('%H:%M:%S')
                print(
                    "[{}|{}]: {}".format(
                        current_time,
                        self.__class__.__name__,
                        message
                    ),
                    end=end
                )
            else:
                print(message, end=end)
        if end != "\n":
            self.do_print_verbose_header = False
        else:
            self.do_print_verbose_header = True
