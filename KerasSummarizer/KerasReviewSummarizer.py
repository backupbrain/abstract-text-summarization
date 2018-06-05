import tensorflow as tf
# import time
from tensorflow.python.layers.core import Dense
import numpy as np
from .zero_state_tensors_patch import _zero_state_tensors
from datetime import datetime


class KerasReviewSummarizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    embeddings_index = None
    word_embedding_matrix = None
    word_vectors = None
    words_to_vectors = None

    def __init__(
        self,
        word_vectors=None,
        words_to_vectors=None,
        embeddings_index_filename=None,
        in_verbose_mode=False
    ):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")
        self.say("Loading tensorflow and numpy...")
        if word_vectors is not None:
            self.load_word_vectors(word_vectors)
        self.__load_embeddings_index(
            embeddings_index_filename,
            words_to_vectors
        )

    def __initialize_model(self):
        '''Create palceholders for inputs to the model'''
        self.say("  Initializing model... ", "")
        input_data = tf.placeholder(
            tf.int32,
            [None, None],
            name='input'
        )
        targets = tf.placeholder(
            tf.int32,
            [None, None],
            name='targets'
        )
        learning_rate = tf.placeholder(
            tf.float32,
            name='learning_rate'
        )
        keep_probability = tf.placeholder(
            tf.float32,
            name='keep_prob'
        )
        summary_length = tf.placeholder(
            tf.int32,
            (None, ),
            name='summary_length'
        )
        max_summary_length = tf.reduce_max(
            summary_length,
            name='max_dec_len'
        )
        text_length = tf.placeholder(
            tf.int32,
            (None, ),
            name='text_length'
        )
        model = {
            "input_data": input_data,
            "targets": targets,
            "learning_rate": learning_rate,
            "keep_probability": keep_probability,
            "summary_length": summary_length,
            "max_summary_length": max_summary_length,
            "text_length": text_length
        }
        self.say("done")
        return model

    def load_word_vectors(self, word_vectors):
        self.word_vectors = word_vectors

    def build_graph(self):
        self.say("Building graph...")
        words_to_vectors = self.words_to_vectors
        # epochs = 100
        batch_size = 64
        rnn_size = 256
        num_layers = 2
        learning_rate = 0.005
        # keep_probability = 0.75

        # Build the graph
        train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with train_graph.as_default():
            model = self.__initialize_model()

            # Create the training and inference logits
            training_logits, inference_logits = self.__seq2seq_model(
                model["targets"],
                model["keep_probability"],
                model["text_length"],
                model["summary_length"],
                model["max_summary_length"],
                tf.reverse(model["input_data"], [-1]),
                len(words_to_vectors)+1,
                rnn_size,
                num_layers,
                words_to_vectors,
                batch_size
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
                optimizer = tf.train.AdamOptimizer(learning_rate)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var)
                    for grad, var in gradients if grad is not None
                ]
                train_op = optimizer.apply_gradients(capped_gradients)
        self.say("Done")

    '''
    def train(self, output_filename):
        # Since I am training this model on my MacBook Pro,
        # it would take me days if I used the whole dataset.
        # For this reason, I am only going to use a subset of the data,
        # so that I can train it over night.
        # Normally I use [FloydHub's](https://www.floydhub.com/)
        # services for my GPU needs, but it would
        # take quite a bit of time to upload the dataset and
        # ConceptNet Numberbatch, so
        # I'm not going to bother with that for this project.
        #
        # I chose not use use the start of the subset because
        # I didn't want to make it too easy for my model.
        # The texts that I am using are closer to the median lengths;
        # I thought this would be more fair.

        # Subset the data for training
        start = 200000
        end = start + 50000
        sorted_summaries_short = sorted_summaries[start:end]
        sorted_texts_short = sorted_texts[start:end]
        self.say("The shortest text length:", len(sorted_texts_short[0]))
        self.say("The longest text length:", len(sorted_texts_short[-1]))

        # Train the Model
        learning_rate_decay = 0.95
        min_learning_rate = 0.0005
        display_step = 20 # Check training loss after every 20 batches
        stop_early = 0
        stop = 3  # If the update loss does not decrease in 3 consecutive update checks, stop training
        per_epoch = 3  # Make 3 update checks per epoch
        update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

        update_loss = 0
        batch_loss = 0
        summary_update_loss = []  # Record the update losses for saving improvements in the model

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If we want to continue training a previous session
            # loader = tf.train.import_meta_graph("./" + output_filename + '.meta')
            # loader.restore(sess, output_filename)

            for epoch_i in range(1, epochs+1):
                update_loss = 0
                batch_loss = 0
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                        get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
                    start_time = time.time()
                    _, loss = sess.run(
                        [train_op, cost],
                        {
                            self.input_data: texts_batch,
                            self.targets: summaries_batch,
                            self.learning_rate: learning_rate,
                            self.summary_length: summaries_lengths,
                            self.text_length: texts_lengths,
                            self.keep_prob: keep_probability
                         }
                    )

                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and batch_i > 0:
                        self.say('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                              .format(epoch_i,
                                      epochs,
                                      batch_i,
                                      len(sorted_texts_short) // batch_size,
                                      batch_loss / display_step,
                                      batch_time*display_step))
                        batch_loss = 0

                    if batch_i % update_check == 0 and batch_i > 0:
                        self.say("Average loss for this update:", round(
                            update_loss/update_check,
                            3
                        ))
                        summary_update_loss.append(update_loss)

                        # If the update loss is at a new minimum,
                        # save the model
                        if update_loss <= min(summary_update_loss):
                            self.say('New Record!')
                            stop_early = 0
                            saver = tf.train.Saver()
                            saver.save(sess, output_filename)

                        else:
                            self.say("No Improvement.")
                            stop_early += 1
                            if stop_early == stop:
                                break
                        update_loss = 0

                # Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                if stop_early == stop:
                    self.say("Stopping Training.")
                    break

    def summarize(self, model_filename):
        # Create your own review or use one from the dataset
        # input_sentence = "I have never eaten an apple before,
        # but this red one was nice. \
        # I think that I will try a green apple next time."
        # text = text_to_seq(input_sentence)
        random = np.random.randint(0, len(clean_texts))
        input_sentence = clean_texts[random]
        text = text_to_seq(clean_texts[random])

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(output_filename + '.meta')
            loader.restore(sess, output_filename)

            self.input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            self.text_length = loaded_graph.get_tensor_by_name('self.text_length:0')
            self.summary_length = loaded_graph.get_tensor_by_name(
                'summary_length:0'
            )
            self.keep_prob = loaded_graph.get_tensor_by_name('self.keep_prob:0')

            # Multiply by batch_size to match the model's input parameters
            answer_logits = sess.run(
                logits,
                {
                    self.input_data: [text]*batch_size,
                    self.summary_length: [np.random.randint(5, 8)],
                    self.text_length: [len(text)]*batch_size,
                    self.keep_prob: 1.0
                }
            )[0]
        # Remove the padding from the tweet
        pad = words_to_vectors["<PAD>"]

        self.say('Original Text:', input_sentence)

        self.say('\nText')
        self.say('  Word Ids:    {}'.format([i for i in text]))
        self.say('  Input Words: {}'.format(" ".join(
            [int_to_vocab[i] for i in text]
        )))

        self.say('\nSummary')
        self.say('  Word Ids:       {}'.format(
            [i for i in answer_logits if i != pad]
        ))
        self.say('  Response Words: {}'.format(" ".join(
            [int_to_vocab[i] for i in answer_logits if i != pad]
        )))

        # Examples of reviews and summaries:
        # - Review(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
        # - Summary(1): great coffee
        #
        #
        # - Review(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!
        # - Summary(2): omg gross gross
        #
        #
        # - Review(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
        # - Summary(3): love it

        # ## Summary

        # I hope that you found this project to be rather interesting and informative. One of my main recommendations for working with this dataset and model is either use a GPU, a subset of the dataset, or plenty of time to train your model. As you might be able to expect, the model will not be able to make good predictions just by seeing many reviews, it needs so see the reviews many times to be able to understand the relationship between words and between descriptions & summaries.
        #
        # In short, I'm pleased with how well this model performs. After creating numerous reviews and checking those from the dataset, I can happily say that most of the generated summaries are appropriate, some of them are great, and some of them make mistakes. I'll try to improve this model and if it gets better, I'll update my GitHub.
        #

    def text_to_seq(self, words_to_vectors, text):
        """Prepare the text for the model"""
        text = clean_text(text)
        result = [
            words_to_vectors.get(word, words_to_vectors['<UNK>'])
            for word in text.split()
        ]
        return result
    '''

    def __load_embeddings_index(
        self,
        embeddings_index_filename,
        words_to_vectors
    ):
        self.words_to_vectors = words_to_vectors
        self.say(
            "Loading embeddings file '{}'... ".format(
                embeddings_index_filename
            ),
            ""
        )
        self.embeddings_index = {}
        with open(embeddings_index_filename) as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = embedding
        self.say("done")
        self.__build_word_embeddings_matrix(
            self.words_to_vectors
        )

    def __build_word_embeddings_matrix(self, words_to_vectors):
        self.say("Building word embeddings matrix... ", "")
        embedding_dim = 300
        num_words = len(words_to_vectors)
        self.word_embedding_matrix = np.zeros(
            (num_words, embedding_dim),
            dtype=np.float32
        )
        for word, i in words_to_vectors.items():
            if word in self.embeddings_index:
                self.word_embedding_matrix[i] = self.embeddings_index[word]
            else:
                # if word is not in CN, cerate a random embedding
                new_embedding = np.array(
                    np.random.uniform(-1.0, 1.0, embedding_dim)
                )
                self.embeddings_index[word] = new_embedding
                self.word_embedding_matrix[i] = new_embedding
        # Check if value matches len(words_to_vectors)
        # print(len(word_embedding_matrix))
        self.say("done")

    def __process_encoding_input(
        self,
        target_data,
        words_to_vectors,
        batch_size
    ):
        '''
        Remove the last word id from each batch and concat
        the <GO> to the begining of each batch
        '''
        self.say("  Processing encoding input... ", "")
        ending = tf.strided_slice(
            target_data,
            [0, 0],
            [batch_size, -1],
            [1, 1]
        )
        dec_input = tf.concat(
            [tf.fill([batch_size, 1], words_to_vectors['<GO>']), ending],
            1
        )
        self.say(" done")
        return dec_input

    def __encoding_layer(
        self,
        rnn_size,
        sequence_length,
        num_layers,
        rnn_inputs,
        keep_probability
    ):
        '''Create the encoding layer'''
        self.say("  Encoding layer... ", "")
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
                    input_keep_prob=keep_probability
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
                    input_keep_prob=keep_probability
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
        self.say("done")
        return enc_output, enc_state

    def __training_decoding_layer(
        self,
        dec_embed_input,
        summary_length,
        dec_cell,
        initial_state,
        output_layer,
        vocab_size,
        max_summary_length
    ):
        """Create the training logits"""
        self.say("  Training encoding layer... ", "")
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=dec_embed_input,
            sequence_length=self.summary_length,
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
            maximum_iterations=self.max_summary_length
        )
        self.say("done")
        return training_logits

    def __inference_decoding_layer(
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
        start_tokens = tf.tile(
            tf.constant(
                [start_token],
                dtype=tf.int32
            ),
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
            maximum_iterations=self.max_summary_length
        )
        return inference_logits

    def __decoding_layer(
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
        words_to_vectors,
        keep_probability,
        batch_size,
        num_layers
    ):
        '''
        Create the decoding cell and attention for the
        training and inference decoding layers
        '''
        self.say("Decoding layer... ")
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
                    input_keep_prob=keep_probability
                )

        output_layer = Dense(
            vocab_size,
            kernel_initializer=tf.truncated_normal_initializer(
                mean=0.0,
                stddev=0.1
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
            training_logits = self.__training_decoding_layer(
                dec_embed_input,
                summary_length,
                dec_cell,
                initial_state,
                output_layer,
                vocab_size,
                max_summary_length
            )
        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.__inference_decoding_layer(
                embeddings,
                words_to_vectors['<GO>'],
                words_to_vectors['<EOS>'],
                dec_cell,
                initial_state,
                output_layer,
                max_summary_length,
                batch_size
            )
        self.say("Done decoding layer")
        return training_logits, inference_logits

    def __seq2seq_model(
        self,
        input_data,
        keep_probability,
        text_length,
        summary_length,
        max_summary_length,
        target_data,
        vocab_size,
        rnn_size,
        num_layers,
        words_to_vectors,
        batch_size,
        word_embedding_matrix
    ):
        '''
        Use the previous functions to
        create the training and inference logits
        '''

        # Use Numberbatch's embeddings and the
        # newly created ones as our embeddings

        self.say("  Building Sequence to Sequence model... ")
        embeddings = self.word_embedding_matrix

        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = self.__encoding_layer(
            rnn_size,
            text_length,
            num_layers,
            enc_embed_input,
            keep_probability
        )
        dec_input = self.__process_encoding_input(
            target_data,
            words_to_vectors,
            batch_size
        )
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

        training_logits, inference_logits = self.__decoding_layer(
            dec_embed_input,
            embeddings,
            enc_output,
            enc_state,
            vocab_size,
            text_length,
            summary_length,
            max_summary_length,
            rnn_size,
            words_to_vectors,
            keep_probability,
            batch_size,
            num_layers
        )
        self.say("Done building Sequence 2 Sequence model")
        return training_logits, inference_logits

    '''
    def __pad_sentence_batch(sentence_batch):
        """
        Pad sentences with <PAD>
        so that each sentence of a batch has the same length
        """
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        result = [
            sentence + [words_to_vectors['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch
        ]
        return result

    def __get_batches(summaries, texts, batch_size):
        """
        Batch summaries, texts, and the lengths of their sentences together
        """
        for batch_i in range(0, len(texts)//batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
            pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))

            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))

            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths
    '''

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
